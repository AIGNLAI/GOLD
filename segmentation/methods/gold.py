import random
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F

from methods.base import TTAMethod 


@torch.no_grad()
def _update_ema_variables(ema_model, model, alpha_teacher):
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha_teacher).add_(param.data * (1.0 - alpha_teacher))
    return ema_model


class GOLDSeg(TTAMethod):

    def __init__(self,model,optimizer,crop_size,steps,episodic,rank=256,tau=0.95,alpha=0.02,T_eig=10,mt=0.999,device="cuda",s_lr=5e-3,s_init_scale=0.0,s_clip=0.5,
                 adapter_scale=0.05,max_pixels_per_batch=512,min_pixels_per_batch=64,n_augmentations=6,rst_m=0.01,ap=0.9):
        super().__init__(model, optimizer, crop_size, steps, episodic)

        self.device = torch.device(device)
        self.rank = int(rank)
        self.tau = float(tau)
        self.alpha = float(alpha)  
        self.T_eig = int(T_eig)
        self.mt = float(mt)      

        self.max_pixels_per_batch = int(max_pixels_per_batch)
        self.min_pixels_per_batch = int(min_pixels_per_batch)

        self.M = None
        self.V = None
        self.S = None
        self._feature_dim = None
        self._s_optimizer = None
        self.step_counter = 0
        self.eps = 1e-12

        self._classifier_layer = None
        self._num_classes = None

        self.s_lr = float(s_lr)
        self.s_init_scale = float(s_init_scale)
        self.s_clip = float(s_clip)
        self.adapter_scale = float(adapter_scale)

        self.n_augmentations = int(n_augmentations)
        self.rst = float(rst_m)
        self.ap = float(ap)

        self.model_ema = deepcopy(self.model)
        for p in self.model_ema.parameters():
            p.detach_()
        self.model_anchor = deepcopy(self.model)
        for p in self.model_anchor.parameters():
            p.detach_()

        self.models = [self.model, self.model_ema, self.model_anchor]
        self.model_states, self.optimizer_state = self.copy_model_and_optimizer()

        scale_ratios = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75]
        self.augmentation_shapes = [
            (int(ratio * 0.5 * crop_size[1]), int(ratio * 0.5 * crop_size[0]))
            for ratio in scale_ratios
        ]

        self.model.to(self.device)
        self.model_ema.to(self.device)
        self.model_anchor.to(self.device)


    def _find_classifier_layer(self, num_classes: int):
        preferred_containers = []
        for name in ["classifier", "head", "decode_head", "segmentation_head"]:
            m = getattr(self.model, name, None)
            if m is not None:
                preferred_containers.append(m)

        def _search(module):
            for _, m in module.named_modules():
                if isinstance(m, nn.Conv2d) and m.out_channels == num_classes:
                    return m
            for _, m in module.named_modules():
                if isinstance(m, nn.Linear) and m.out_features == num_classes:
                    return m
            return None

        for container in preferred_containers:
            hit = _search(container)
            if hit is not None:
                return hit

        hit = _search(self.model)
        if hit is not None:
            return hit

        raise RuntimeError(f"Cannot find classifier layer with out_channels/out_features == {num_classes}")

    @torch.no_grad()
    def _get_classifier_W(self, num_classes: int) -> torch.Tensor:
        if self._classifier_layer is None:
            self._classifier_layer = self._find_classifier_layer(num_classes)

        layer = self._classifier_layer
        if isinstance(layer, nn.Conv2d):
            W = layer.weight  # (C, L, kh, kw)
            if W.shape[0] != num_classes:
                raise RuntimeError("Classifier out_channels mismatch.")
            if W.shape[2:] == (1, 1):
                return W.squeeze(-1).squeeze(-1).contiguous()  # (C,L)
            # if not 1x1, average spatially as proxy
            return W.mean(dim=(2, 3)).contiguous()
        elif isinstance(layer, nn.Linear):
            W = layer.weight  # (C,L)
            if W.shape[0] != num_classes:
                raise RuntimeError("Classifier out_features mismatch.")
            return W.contiguous()
        else:
            raise RuntimeError(f"Unsupported classifier type: {type(layer)}")


    def _forward_capture_precls(self, x):
        """
        Returns:
          feat_precls: (B, L, Hf, Wf) - input to classifier layer
          logits:      (B, C, H, W)   - model(x) output if tensor else classifier output
        """
        if self._classifier_layer is None:
            # infer C from a forward
            y = self.model(x)
            if not (isinstance(y, torch.Tensor) and y.ndim == 4):
                raise RuntimeError("Expected segmentation logits (B,C,H,W) from model(x).")
            C = int(y.shape[1])
            self._num_classes = C
            self._classifier_layer = self._find_classifier_layer(C)

        holder = {}

        def _hook(module, inp, out):
            holder["feat"] = inp[0]
            holder["logits_head"] = out

        h = self._classifier_layer.register_forward_hook(_hook)
        logits = self.model(x)
        h.remove()

        if "feat" not in holder:
            raise RuntimeError("Hook did not capture pre-classifier features.")

        feat_precls = holder["feat"]
        if isinstance(logits, torch.Tensor) and logits.ndim == 4:
            logits_out = logits
        else:
            logits_out = holder["logits_head"]

        return feat_precls, logits_out


    def _lazy_init(self, feat_precls, logits):
        if feat_precls.ndim != 4:
            raise ValueError("Expected pre-classifier features [B,L,H,W]")
        if logits.ndim != 4:
            raise ValueError("Expected logits [B,C,H,W]")

        L = int(feat_precls.shape[1])
        C = int(logits.shape[1])
        self._feature_dim = L
        self._num_classes = C

        if self._classifier_layer is None:
            self._classifier_layer = self._find_classifier_layer(C)

        # Warm start: M0 = W^T W
        W = self._get_classifier_W(C).to(self.device).to(feat_precls.dtype)  # (C,L)
        self.M = W.t().matmul(W).contiguous()  # (L,L)

        # Init V from eig(M0)
        self._update_subspace()

        # Init S
        r = min(self.rank, L)
        if self.s_init_scale != 0.0:
            init = self.s_init_scale * torch.randn(r, device=self.device, dtype=feat_precls.dtype)
        else:
            init = torch.zeros(r, device=self.device, dtype=feat_precls.dtype)
        self.S = nn.Parameter(init)

        # S optimizer (separate LR)
        params = [self.S]
        self._s_optimizer = torch.optim.SGD(params, lr=self.s_lr, momentum=0.9)


    def _apply_adapter(self, features):
        """
        features: B x L x H x W  (pre-classifier features)
        f' = f + adapter_scale * V @ ( S * (V^T f) )
        """
        V = self.V.to(features.device)  # L x r
        S = self.S
        # u = V^T f -> B x r x H x W
        u = torch.einsum("blhw,lr->brhw", features, V)
        su = S.view(1, -1, 1, 1) * u
        delta = torch.einsum("lr,brhw->blhw", V, su)
        return features + (self.adapter_scale * delta)

    def _logits_from_features(self, feat_precls):
        if self._classifier_layer is None:
            assert self._num_classes is not None
            self._classifier_layer = self._find_classifier_layer(self._num_classes)
        layer = self._classifier_layer
        try:
            return layer(feat_precls)
        except Exception:
            return self.model(feat_precls)

    def _update_subspace(self):
        Msym = 0.5 * (self.M + self.M.t())
        w, Q = torch.linalg.eigh(Msym)
        r = min(self.rank, Msym.shape[0])
        idx = torch.argsort(w, descending=True)[:r]
        self.V = Q[:, idx].contiguous().to(self.device)

    def _sample_mask_indices(self, mask_flat: torch.Tensor, max_k: int, min_k: int):
        idx = torch.nonzero(mask_flat, as_tuple=False).squeeze(1)
        if idx.numel() < min_k:
            return None
        if idx.numel() > max_k:
            perm = torch.randperm(idx.numel(), device=idx.device)[:max_k]
            idx = idx[perm]
        return idx

    def _compute_agop_batch(self, feat_precls, logits_teacher):
        """
        feat_precls: (B,L,Hf,Wf) requires_grad_(True)
        logits_teacher: (B,C,Ht,Wt) from teacher (outputs_ema), used for mask & c*(p).
        Returns: M_batch (L,L) or None
        """
        B, L, Hf, Wf = feat_precls.shape

        # teacher probs at feature resolution
        with torch.no_grad():
            probs_t = F.softmax(logits_teacher, dim=1)
            if probs_t.shape[2:] != (Hf, Wf):
                probs_t = F.interpolate(probs_t, size=(Hf, Wf), mode="bilinear", align_corners=True)
            conf, pred = probs_t.max(dim=1)  # (B,Hf,Wf)
            mask = conf >= self.tau

            mask_flat = mask.view(-1)
            idx = self._sample_mask_indices(mask_flat, self.max_pixels_per_batch, self.min_pixels_per_batch)
            if idx is None:
                return None

            pred_flat = pred.view(-1)[idx]  # (K,)
            # map idx -> (b,y,x)
            b = idx // (Hf * Wf)
            rem = idx % (Hf * Wf)
            yy = rem // Wf
            xx = rem % Wf

        logits_s = self._logits_from_features(feat_precls)
        if logits_s.shape[2:] != (Hf, Wf):
            logits_s = F.interpolate(logits_s, size=(Hf, Wf), mode="bilinear", align_corners=True)

        s_sum = torch.zeros((), device=feat_precls.device, dtype=logits_s.dtype)
        K = int(pred_flat.numel())
        for i in range(K):
            s_sum = s_sum + logits_s[b[i], pred_flat[i], yy[i], xx[i]]

        g_full = torch.autograd.grad(
            s_sum, feat_precls,
            retain_graph=True,
            create_graph=False,
            allow_unused=False
        )[0]  # (B,L,Hf,Wf)

        M_batch = feat_precls.new_zeros((L, L))
        for i in range(K):
            g = g_full[b[i], :, yy[i], xx[i]]  # (L,)
            M_batch += torch.ger(g, g)

        M_batch = M_batch / float(max(K, 1))
        return M_batch.detach()

    @torch.no_grad()
    def create_ensemble_pred(self, x, ema_pred):
        inp_shape = x.shape[2:]
        for aug_shape in self.augmentation_shapes:
            flip = [random.random() <= 0.5 for _ in range(x.shape[0])]
            tmp_input = torch.cat(
                [x[i:i+1].flip(dims=(3,)) if fp else x[i:i+1] for i, fp in enumerate(flip)], dim=0
            )
            tmp_input = F.interpolate(tmp_input, size=aug_shape, mode="bilinear", align_corners=True)

            try:
                tmp_output = self.model_ema(tmp_input)
            except Exception:
                tmp_output = self.model_ema([tmp_input, False])

            tmp_output = torch.cat(
                [tmp_output[i:i+1].flip(dims=(3,)) if fp else tmp_output[i:i+1] for i, fp in enumerate(flip)], dim=0
            )
            ema_pred = ema_pred + F.interpolate(tmp_output, size=inp_shape, mode="bilinear", align_corners=True)

        ema_pred /= (len(self.augmentation_shapes) + 1)
        return ema_pred

    def _stochastic_restore(self):
        if self.rst <= 0.0:
            return
        for nm, m in self.model.named_modules():
            for npp, p in m.named_parameters(recurse=False):
                if npp in ["weight", "bias"] and p.requires_grad:
                    mask = (torch.rand(p.shape, device=p.device) < self.rst).float()
                    with torch.no_grad():
                        key = f"{nm}.{npp}"
                        if key in self.model_states[0]:
                            p.data = self.model_states[0][key].to(p.device) * mask + p.data * (1.0 - mask)

    @torch.enable_grad()
    def forward_and_adapt(self, x):
        outputs = self.model(x)

        anchor_prob = torch.softmax(self.model_anchor(x), dim=1).max(dim=1)[0]
        outputs_ema = self.model_ema(x)

        if anchor_prob.mean() < self.ap:
            outputs_ema = self.create_ensemble_pred(x, outputs_ema)

        loss_student = (-(outputs_ema.softmax(1) * outputs.log_softmax(1)).sum(1)).mean()
        self.optimizer.zero_grad(set_to_none=True)
        loss_student.backward()
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)

        with torch.no_grad():
            _update_ema_variables(self.model_ema, self.model, self.mt)

        self._stochastic_restore()

        feat_precls, logits_base = self._forward_capture_precls(x)
        feat_precls = feat_precls.to(self.device)
        logits_base = logits_base.to(self.device)

        if self._feature_dim is None:
            self._lazy_init(feat_precls.detach(), logits_base.detach())

        feat_precls = feat_precls.detach().requires_grad_(True)

        M_batch = self._compute_agop_batch(feat_precls, outputs_ema.detach())
        if M_batch is not None:
            with torch.no_grad():
                self.M = (1.0 - self.alpha) * self.M + self.alpha * M_batch.to(self.device)

        self.step_counter += 1
        if (self.step_counter % max(1, self.T_eig) == 0) and (self.M is not None):
            with torch.no_grad():
                self._update_subspace()

        features_adapt = self._apply_adapter(feat_precls)
        logits_adapt = self._logits_from_features(features_adapt)

        probs_ema = F.softmax(outputs_ema.detach(), dim=1)
        probs_adapt = F.softmax(logits_adapt, dim=1)
        if probs_adapt.shape[2:] != probs_ema.shape[2:]:
            probs_adapt = F.interpolate(probs_adapt, size=probs_ema.shape[2:], mode="bilinear", align_corners=True)

        loss_agop = (-(probs_ema * probs_adapt.log()).sum(1)).mean()

        self._s_optimizer.zero_grad(set_to_none=True)
        loss_agop.backward()
        if self.S.grad is not None:
            torch.nn.utils.clip_grad_norm_([self.S], max_norm=1.0)
        self._s_optimizer.step()
        with torch.no_grad():
            self.S.clamp_(min=-self.s_clip, max=self.s_clip)

        return outputs_ema

    def forward(self, x):
        if self.episodic:
            self.reset()

        for _ in range(self.steps):
            x_new = torch.cat([self.rand_crop(x.clone()) for _ in range(2)], dim=0)
            _ = self.forward_and_adapt(x_new)

        return self.model_ema(x.to(self.device))