import logging
import os
import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F

from methods.base import TTAMethod
from models.model import split_up_model
from augmentations.transforms_cotta import get_tta_transforms
from datasets.data_loading import get_source_loader
from utils.registry import ADAPTATION_REGISTRY
from utils.losses import SymmetricCrossEntropy
from utils.misc import ema_update_model

logger = logging.getLogger(__name__)


@ADAPTATION_REGISTRY.register()
class GOLD(TTAMethod):
    """
    
    """

    def __init__(self, cfg, model, num_classes):
        super().__init__(cfg, model, num_classes)

        batch_size_src = cfg.TEST.BATCH_SIZE if cfg.TEST.BATCH_SIZE > 1 else cfg.TEST.WINDOW_LENGTH
        _, self.src_loader = get_source_loader(dataset_name=cfg.CORRUPTION.DATASET,
                                               adaptation=cfg.MODEL.ADAPTATION,
                                               preprocess=model.model_preprocess,
                                               data_root_dir=cfg.DATA_DIR,
                                               batch_size=batch_size_src,
                                               ckpt_path=cfg.MODEL.CKPT_PATH,
                                               num_samples=cfg.SOURCE.NUM_SAMPLES,
                                               percentage=cfg.SOURCE.PERCENTAGE,
                                               workers=min(cfg.SOURCE.NUM_WORKERS, os.cpu_count()),use_clip=False)
        
        self.src_loader_iter = iter(self.src_loader)
        self.contrast_mode = cfg.CONTRAST.MODE
        self.temperature = cfg.CONTRAST.TEMPERATURE
        self.base_temperature = self.temperature
        self.projection_dim = cfg.CONTRAST.PROJECTION_DIM
        self.lambda_ce_trg = cfg.GOLD.LAMBDA_CE_TRG
        self.lambda_cont = cfg.GOLD.LAMBDA_CONT
        self.m_teacher_momentum = cfg.M_TEACHER.MOMENTUM
        self.warmup_steps = cfg.GOLD.NUM_SAMPLES_WARM_UP // batch_size_src
        self.final_lr = cfg.OPTIM.LR
        arch_name = cfg.MODEL.ARCH
        ckpt_path = cfg.MODEL.CKPT_PATH

        self.tta_transform = get_tta_transforms(self.img_size)

        # setup loss functions
        self.symmetric_cross_entropy = SymmetricCrossEntropy()

        # EMA teacher
        self.model_ema = self.copy_model(self.model)
        for param in self.model_ema.parameters():
            param.detach_()

        # split model
        self.feature_extractor, self.classifier = split_up_model(self.model, arch_name, self.dataset_name)

        proto_dir_path = os.path.join(cfg.CKPT_DIR, "prototypes")
        if self.dataset_name == "domainnet126":
            fname = f"protos_{self.dataset_name}_{ckpt_path.split(os.sep)[-1].split('_')[1]}.pth"
        else:
            fname = f"protos_{self.dataset_name}_{arch_name}.pth"
        fname = os.path.join(proto_dir_path, fname)

        if os.path.exists(fname):
            logger.info("Loading class-wise source prototypes...")
            self.prototypes_src = torch.load(fname)
        else:
            os.makedirs(proto_dir_path, exist_ok=True)
            features_src = torch.tensor([])
            labels_src = torch.tensor([])
            logger.info("Extracting source prototypes...")
            with torch.no_grad():
                for data in tqdm.tqdm(self.src_loader):
                    x, y = data[0], data[1]
                    tmp_features = self.feature_extractor(x.to(self.device))
                    if tmp_features.dim() == 3:
                        tmp_features = tmp_features.view(tmp_features.shape[0], -1)
                    features_src = torch.cat([features_src, tmp_features.view(tmp_features.shape[0], -1).cpu()], dim=0) \
                        if features_src.numel() else tmp_features.view(tmp_features.shape[0], -1).cpu()
                    labels_src = torch.cat([labels_src, y], dim=0) if labels_src.numel() else y

            self.prototypes_src = torch.tensor([])
            for i in range(self.num_classes):
                mask = labels_src == i
                if mask.sum() == 0:
                    if self.prototypes_src.numel() == 0:
                        self.prototypes_src = torch.zeros((1, features_src.shape[1]))
                    self.prototypes_src = torch.cat([self.prototypes_src, torch.zeros((1, features_src.shape[1]))], dim=0)
                else:
                    class_proto = features_src[mask].mean(dim=0, keepdim=True)
                    if self.prototypes_src.numel() == 0:
                        self.prototypes_src = class_proto
                    else:
                        self.prototypes_src = torch.cat([self.prototypes_src, class_proto], dim=0)
            torch.save(self.prototypes_src, fname)

        # move prototypes to device and shape to (num_classes, 1, L)
        self.prototypes_src = self.prototypes_src.to(self.device).unsqueeze(1)
        self.prototype_labels_src = torch.arange(start=0, end=self.num_classes, step=1).to(self.device).long()

        # projector (contrastive head)
        if self.dataset_name == "domainnet126":
            self.projector = nn.Identity()
        else:
            num_channels = self.prototypes_src.shape[-1]
            self.projector = nn.Sequential(nn.Linear(num_channels, self.projection_dim), nn.ReLU(),
                                           nn.Linear(self.projection_dim, self.projection_dim)).to(self.device)
            self.optimizer.add_param_group({'params': self.projector.parameters(), 'lr': self.optimizer.param_groups[0]["lr"]})

        self.alpha = cfg.GOLD.ALPHA
        self.top_r = cfg.GOLD.TOP_R 
        self.conf_thresh = cfg.GOLD.CONF_THRESH
        self.update_every = cfg.GOLD.UPDATE_EVERY 
        self.adapter_lr_scale = cfg.GOLD.ADAPTER_LR_SCALE

        self.feature_dim = int(self.prototypes_src.shape[-1])

        logger.info(f"[GOLD] feature_dim={self.feature_dim}, top_r={self.top_r}, "
                    f"alpha={self.alpha}, conf_thresh={self.conf_thresh}, update_every={self.update_every}")
        
        W = self.classifier.weight.detach().clone()
        
        self.M_online = (W.T @ W).to(self.device)
        self._update_eig()
        self.adapter_S = nn.Parameter(torch.zeros(self.top_r, device=self.device))
        base_lr = self.optimizer.param_groups[0]["lr"]
        self.optimizer.add_param_group({'params': [self.adapter_S], 'lr': base_lr * float(self.adapter_lr_scale)})
        self._batch_counter = 0

        if self.warmup_steps > 0:
            warmup_ckpt_path = os.path.join(cfg.CKPT_DIR, "warmup")
            if self.dataset_name == "domainnet126":
                source_domain = ckpt_path.split(os.sep)[-1].split('_')[1]
                ckpt_path = f"ckpt_warmup_{self.dataset_name}_{source_domain}_{arch_name}_bs{self.src_loader.batch_size}.pth"
            else:
                ckpt_path = f"ckpt_warmup_{self.dataset_name}_{arch_name}_{cfg.MODEL.ADAPTATION}_bs{self.src_loader.batch_size}.pth"
            ckpt_path = os.path.join(warmup_ckpt_path, ckpt_path)

            if os.path.exists(ckpt_path):
                logger.info("Loading warmup checkpoint...")
                checkpoint = torch.load(ckpt_path)
                self.model.load_state_dict(checkpoint["model"])
                self.model_ema.load_state_dict(checkpoint["model_ema"])
                self.optimizer.load_state_dict(checkpoint["optimizer"])
                logger.info(f"Loaded from {ckpt_path}")
            else:
                os.makedirs(warmup_ckpt_path, exist_ok=True)
                self.warmup()
                torch.save({"model": self.model.state_dict(),
                            "model_ema": self.model_ema.state_dict(),
                            "optimizer": self.optimizer.state_dict()
                            }, ckpt_path)

        self.models = [self.model, self.model_ema, self.projector]
        self.model_states, self.optimizer_state = self.copy_model_and_optimizer()

    def _compute_batch_agop(self, features, logits):
        probs = F.softmax(logits, dim=1)
        maxp, _ = probs.max(dim=1)
        mask = maxp >= float(self.conf_thresh)
        if mask.sum() < 2:
            return None
        f_mask = features[mask] 
        f_det = f_mask.detach().requires_grad_(True)
        logits_det = self.classifier(f_det) 
        top_vals, _ = logits_det.max(dim=1)
        grads = torch.autograd.grad(top_vals.sum(), f_det, retain_graph=False, create_graph=False)[0] 
        batch_agop = grads.T @ grads
        batch_agop = batch_agop / float(grads.shape[0])
        del f_det, logits_det, top_vals, grads
        torch.cuda.empty_cache()
        return batch_agop

    def _update_M_online(self, batch_agop):
        self.M_online = (1.0 - float(self.alpha)) * self.M_online + float(self.alpha) * batch_agop
        
    def collect_params(self):
        params = [self.adapter_S]
        names = ['scale_adpater']

        if self.params == 'full':
            for nm, m in self.model.named_modules():
                for np, p in m.named_parameters():
                    if np in ['weight', 'bias'] and p.requires_grad:
                        params.append(p)
                        names.append(f"{nm}.{np}")
        else:
            for nm, m in self.model.named_modules():
                if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.LayerNorm, nn.GroupNorm)):
                    for np, p in m.named_parameters():
                        if np in ['weight', 'bias']:
                            params.append(p)
                            names.append(f"{nm}.{np}")
        return params, names

    def _update_eig(self):
        device = self.M_online.device
        vals, vecs = torch.linalg.eigh(self.M_online.to(device))
        r = min(self.top_r, vals.numel())
        idx = torch.argsort(vals, descending=True)[:r]
        top_vecs = vecs[:, idx]
        self.V = top_vecs

    def apply_adapter(self, features):
        f_proj = features @ self.V  # (B, r)
        scale = (1.0 + self.adapter_S).view(1, -1)
        f_adapt_proj = f_proj * scale
        delta = (f_adapt_proj - f_proj) @ (self.V.T)  # (B, L)
        return features + delta

    @torch.enable_grad()
    def warmup(self):
        logger.info("Starting warm up...")
        for i in range(self.warmup_steps):
            try:
                batch = next(self.src_loader_iter)
            except StopIteration:
                self.src_loader_iter = iter(self.src_loader)
                batch = next(self.src_loader_iter)

            imgs_src = batch[0].to(self.device)

            outputs = self.model(imgs_src)
            outputs_ema = self.model_ema(imgs_src)

            loss = self.symmetric_cross_entropy(outputs, outputs_ema).mean(0)
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

            self.model_ema = ema_update_model(
                model_to_update=self.model_ema,
                model_to_merge=self.model,
                momentum=self.m_teacher_momentum,
                device=self.device,
                update_all=True
            )
        logger.info("Finished warm up...")
    

    def contrastive_loss(self, features, labels=None, mask=None):
        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(self.device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(self.device)
        else:
            mask = mask.float().to(self.device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        contrast_feature = self.projector(contrast_feature)
        contrast_feature = F.normalize(contrast_feature, p=2, dim=1)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(torch.matmul(anchor_feature, contrast_feature.T), self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(self.device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()
        return loss


    def loss_calculation(self, x):
        imgs_test = x[0]
        features_test = self.feature_extractor(imgs_test)
        features_test_adapt = self.apply_adapter(features_test)
        outputs_test = self.classifier(features_test_adapt)

        features_aug_test = self.feature_extractor(self.tta_transform(imgs_test))
        features_aug_test_adapt = self.apply_adapter(features_aug_test)
        outputs_aug_test = self.classifier(features_aug_test_adapt)

        outputs_ema = self.model_ema(imgs_test)

        batch_agop = self._compute_batch_agop(features_test.detach(), outputs_test.detach())
        if batch_agop is not None:
            self._update_M_online(batch_agop)
        self._batch_counter += 1
        if (self._batch_counter % max(1, int(self.update_every))) == 0:
            self._update_eig()
        
        with torch.no_grad():
            dist = F.cosine_similarity(
                x1=self.prototypes_src.repeat(1, features_test_adapt.shape[0], 1),
                x2=features_test_adapt.view(
                    1, features_test_adapt.shape[0], features_test_adapt.shape[1]
                ).repeat(self.prototypes_src.shape[0], 1, 1),
                dim=-1
            )
            _, indices = dist.topk(1, largest=True, dim=0)
            indices = indices.squeeze(0)

        features_for_contrast = torch.cat(
            [
                self.prototypes_src[indices],  # (B,1,L) prototype anchor
                features_test_adapt.view(features_test_adapt.shape[0], 1, features_test_adapt.shape[1]),
                features_aug_test_adapt.view(features_aug_test_adapt.shape[0], 1, features_aug_test_adapt.shape[1]),
            ],
            dim=1
        )
        loss_contrastive = self.contrastive_loss(features=features_for_contrast, labels=None)

        loss_self_training = (0.5 * self.symmetric_cross_entropy(outputs_test, outputs_ema) +
                              0.5 * self.symmetric_cross_entropy(outputs_aug_test, outputs_ema)).mean(0)
        loss = self.lambda_ce_trg * loss_self_training + self.lambda_cont * loss_contrastive

        outputs = outputs_test + outputs_ema
        return outputs, loss

    @torch.enable_grad()
    def forward_and_adapt(self, x):
        if self.mixed_precision and self.device == "cuda":
            with torch.cuda.amp.autocast():
                outputs, loss = self.loss_calculation(x)
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad()
        else:
            outputs, loss = self.loss_calculation(x)
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

        self.model_ema = ema_update_model(
            model_to_update=self.model_ema,
            model_to_merge=self.model,
            momentum=self.m_teacher_momentum,
            device=self.device,
            update_all=True
        )
        return outputs

    @torch.no_grad()
    def forward_sliding_window(self, x):
        imgs_test = x[0]
        features_test = self.feature_extractor(imgs_test)
        features_test_adapt = self.apply_adapter(features_test)
        outputs_test = self.classifier(features_test_adapt)
        outputs_ema = self.model_ema(imgs_test)
        return outputs_test + outputs_ema

    def configure_model(self):
        self.model.eval()
        self.model.requires_grad_(False)
        for m in self.model.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.requires_grad_(True)
                m.track_running_stats = False
                m.running_mean = None
                m.running_var = None
            elif isinstance(m, nn.BatchNorm1d):
                m.train()
                m.requires_grad_(True)
            else:
                m.requires_grad_(True)
