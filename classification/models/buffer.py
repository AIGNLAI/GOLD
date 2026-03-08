import numpy as np
import torch
import torch.nn as nn


class Buffer(nn.Module):
    def __init__(self, args, input_size=None, n_classes=100):
        super().__init__()
        self.args = args
        self.place_left = True

        if input_size is None:
            input_size = args.input_size

        buffer_size = args.OBAO.BUFFER_SIZE
        self.pass_by = 0
        self.buffer_size = buffer_size
        print('buffer has %d slots' % buffer_size)

        bx = torch.FloatTensor(buffer_size, *input_size).fill_(0)
        print("bx", bx.shape)
        by = torch.LongTensor(buffer_size).fill_(0)

        logits = torch.FloatTensor(buffer_size, n_classes).fill_(0)
        ents = torch.FloatTensor(buffer_size).fill_(0)

        bx = bx.cuda()
        by = by.cuda()
        logits = logits.cuda()

        self.current_index = 0
        self.n_seen_so_far = 0
        self.is_full = 0

        # registering as buffer allows us to save the object using `torch.save`
        self.register_buffer('bx', bx)
        self.register_buffer('by', by)
        self.register_buffer('logits', logits)
        self.register_buffer('ents', ents)

        self.to_one_hot = lambda x: x.new(x.size(0), args.n_classes).fill_(0).scatter_(1, x.unsqueeze(1), 1)
        self.arange_like = lambda x: torch.arange(x.size(0)).to(x.device)
        self.shuffle = lambda x: x[torch.randperm(x.size(0))]

        self.seen_classes = set()
        self.all_classes = set(range(n_classes))
        self.unseen = set()
        self.seen_per_batch = []

    @property
    def x(self):
        return self.bx[:self.current_index]

    @property
    def y(self):
        return self.to_one_hot(self.by[:self.current_index])

    def logits_mem(self):
        num = min(self.current_index, self.buffer_size)
        return self.logits[:num]

    def add_reservoir(self, x, y, ents, logits):
        n_elem = x.size(0)
        # add whatever still fits in the buffer
        place_left = max(0, self.bx.size(0) - self.current_index)
        if place_left:
            offset = min(place_left, n_elem)
            self.bx[self.current_index: self.current_index + offset].data.copy_(x[:offset])
            self.by[self.current_index: self.current_index + offset].data.copy_(y[:offset])
            self.ents[self.current_index: self.current_index + offset].data.copy_(ents[:offset])
            self.logits[self.current_index: self.current_index + offset].data.copy_(logits[:offset])

            self.current_index += offset
            self.n_seen_so_far += offset

            # everything was added
            if offset == x.size(0):
                return

        self.place_left = False

        # remove what is already in the buffer
        x, y, ents, logits = x[place_left:], y[place_left:], ents[place_left:], \
                                      logits[place_left:]

        indices = torch.FloatTensor(x.size(0)).to(x.device).uniform_(0, self.n_seen_so_far).long()
        valid_indices = (indices < self.bx.size(0)).long()

        idx_new_data = valid_indices.nonzero().squeeze(-1)
        idx_buffer = indices[idx_new_data]

        self.n_seen_so_far += x.size(0)

        if idx_buffer.numel() == 0:
            return

        assert idx_buffer.max() < self.bx.size(0)
        assert idx_buffer.max() < self.by.size(0)
        assert idx_buffer.max() < self.ents.size(0)

        assert idx_new_data.max() < x.size(0)
        assert idx_new_data.max() < y.size(0)
        assert idx_new_data.max() < ents.size(0)

        # perform overwrite op
        self.bx[idx_buffer] = x[idx_new_data].cuda()
        self.by[idx_buffer] = y[idx_new_data].cuda()
        self.ents[idx_buffer] = ents[idx_new_data].cuda()
        self.logits[idx_buffer] = logits[idx_new_data].cuda()

        return idx_buffer

    def add_reservoir_ent(self, x, y, ents, logits):
        n_elem = x.size(0)
        # add whatever still fits in the buffer
        place_left = max(0, self.bx.size(0) - self.current_index)
        if place_left:
            offset = min(place_left, n_elem)
            self.bx[self.current_index: self.current_index + offset].data.copy_(x[:offset])
            self.by[self.current_index: self.current_index + offset].data.copy_(y[:offset])
            self.ents[self.current_index: self.current_index + offset].data.copy_(ents[:offset])
            self.logits[self.current_index: self.current_index + offset].data.copy_(logits[:offset])

            self.current_index += offset
            self.n_seen_so_far += offset

            # everything was added
            if offset == x.size(0):
                return

        self.place_left = False

        # remove what is already in the buffer
        x, y, ents, logits = x[place_left:], y[place_left:], ents[place_left:], \
                                      logits[place_left:]

        self.n_seen_so_far += x.size(0)

        # sorted by ents
        sorted_indices = torch.argsort(self.ents, descending=True)
        num_to_replace = x.size(0)

        idx_buffer = sorted_indices[:num_to_replace]
        valid_indices = (idx_buffer < self.bx.size(0)).long()
        idx_new_data = valid_indices.nonzero().squeeze(-1)

        if idx_buffer.numel() == 0:
            return

        assert idx_buffer.max() < self.bx.size(0)
        assert idx_buffer.max() < self.by.size(0)

        assert idx_new_data.max() < x.size(0)
        assert idx_new_data.max() < y.size(0)

        # perform overwrite op
        self.bx[idx_buffer] = x[idx_new_data].cuda()
        self.by[idx_buffer] = y[idx_new_data].cuda()
        self.ents[idx_buffer] = ents[idx_new_data].cuda()
        self.logits[idx_buffer] = logits[idx_new_data].cuda()

        y_cur = self.by.cpu().numpy()
        self.seen_classes = self.seen_classes | set(y_cur)

        self.unseen = self.all_classes - self.seen_classes
        return idx_buffer

    def sample(self, amt, exclude_task=None, ret_ind=False):
        if self.n_seen_so_far == 0 or self.n_seen_so_far < self.pass_by:
            return None, None, None, None

        if exclude_task is not None:
            valid_indices = (self.t != exclude_task)
            valid_indices = valid_indices.nonzero().squeeze()
            bx, by, ents, logits = self.bx[valid_indices], self.by[valid_indices], self.ents[valid_indices], \
                                 self.logits[valid_indices]
        else:
            bx, by, ents, logits = self.bx[:self.current_index], self.by[:self.current_index], \
                self.ents[:self.current_index], self.logits[:self.current_index]

        if bx.size(0) < amt:
            if ret_ind:
                return bx, by, ents, logits, torch.from_numpy(np.arange(bx.size(0)))
            else:
                return bx, by, ents, logits
        else:
            indices = torch.from_numpy(np.random.choice(bx.size(0), amt, replace=False))

            indices = indices.cuda()

            if ret_ind:
                return bx[indices], by[indices], ents[indices], logits[indices], indices
            else:
                return bx[indices], by[indices], ents[indices], logits[indices]