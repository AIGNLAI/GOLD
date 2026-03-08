import logging
import numpy as np
import os
import tqdm
import torch
import torch.nn as nn
import torch.jit
import torch.nn.functional as F
import math
from methods.base import TTAMethod
from models.model import split_up_model
from augmentations.transforms_cotta import get_tta_transforms
from datasets.data_loading import get_source_loader
from utils.registry import ADAPTATION_REGISTRY
from models.buffer import Buffer
from utils.losses import SymmetricCrossEntropy

logger = logging.getLogger(__name__)


def update_ema_variables(ema_model, model, alpha_teacher):
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data[:] = alpha_teacher * ema_param[:].data[:] + (1 - alpha_teacher) * param[:].data[:]
    return ema_model


@ADAPTATION_REGISTRY.register()
class OBAO(TTAMethod):
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
                                               workers=min(cfg.SOURCE.NUM_WORKERS, os.cpu_count()),
                                               use_clip=False)
        self.src_loader_iter = iter(self.src_loader)

        self.lambda_ce_src = cfg.OBAO.LAMBDA_CE_SRC
        self.lambda_ce_trg = cfg.OBAO.LAMBDA_CE_TRG
        self.m_teacher_momentum = cfg.M_TEACHER.MOMENTUM
        self.projection_dim = cfg.CONTRAST.PROJECTION_DIM
        # arguments neeeded for warm up
        self.warmup_steps = cfg.OBAO.NUM_SAMPLES_WARM_UP // batch_size_src
        self.final_lr = cfg.OPTIM.LR
        arch_name = cfg.MODEL.ARCH
        ckpt_path = cfg.MODEL.CKPT_PATH
        self.rst = cfg.OBAO.RST
        self.tta_transform = get_tta_transforms(self.img_size)

        # setup loss functions
        self.symmetric_cross_entropy = SymmetricCrossEntropy()

        # setup buffer setting
        self.lambda_ce_low_ent = cfg.OBAO.LAMBDA_CE_LOW_ENT
        self.num_samples_low_ent = 0
        self.e_margin = math.log(self.num_classes) * cfg.OBAO.E_MARGIN

        if "cifar" in self.dataset_name:
            input_size = [3, 32, 32]
        else:
            input_size = [3, 224, 224]
        self.buffer = Buffer(self.cfg, input_size, self.num_classes).cuda()
        self.buffer_batch_size = cfg.OBAO.BUFFER_BS_SIZE
        print(f"buffer batch size {self.buffer_batch_size} ")

        # setup c-r tranfer
        self.lambda_CRP = cfg.OBAO.LAMBDA_CRP
        self.org_class_relation_form = cfg.OBAO.ORG_CLASS_RELATION_FORM
        self.class_relation_type = cfg.OBAO.CLASS_RELATION_TYPE
        self.class_relation_loss_type = cfg.OBAO.CLASS_RELATION_LOSS_TYPE
        self.non_diag_alpha = 1.0

        # Setup EMA model
        self.model_ema = self.copy_model(self.model)
        for param in self.model_ema.parameters():
            param.detach_()

        # split up the model
        self.feature_extractor, self.classifier = split_up_model(self.model, arch_name, self.dataset_name)
        self.feature_extractor_ema, self.classifier_ema = split_up_model(self.model_ema, arch_name, self.dataset_name)

        # define the prototype paths
        proto_dir_path = os.path.join(cfg.CKPT_DIR, "prototypes")
        if self.dataset_name == "domainnet126":
            fname = f"protos_{self.dataset_name}_{ckpt_path.split(os.sep)[-1].split('_')[1]}.pth"
        else:
            fname = f"protos_{self.dataset_name}_{arch_name}.pth"
        fname = os.path.join(proto_dir_path, fname)

        # get source prototypes
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
                    features_src = torch.cat([features_src, tmp_features.view(tmp_features.shape[:2]).cpu()], dim=0)
                    labels_src = torch.cat([labels_src, y], dim=0)
                    if len(features_src) > 100000:
                        break

            # create class-wise source prototypes
            self.prototypes_src = torch.tensor([])
            for i in range(self.num_classes):
                mask = labels_src == i
                self.prototypes_src = torch.cat([self.prototypes_src, features_src[mask].mean(dim=0, keepdim=True)], dim=0)

            torch.save(self.prototypes_src, fname)

        self.prototypes_src = self.prototypes_src.to(self.device).unsqueeze(1)
        self.prototype_labels_src = torch.arange(start=0, end=self.num_classes, step=1).to(self.device).long()

        # setup projector
        if self.dataset_name == "domainnet126":
            # do not use a projector since the network already clusters the features and reduces the dimensions
            self.projector = nn.Identity()
        else:
            num_channels = self.prototypes_src.shape[-1]
            self.projector = nn.Sequential(nn.Linear(num_channels, self.projection_dim), nn.ReLU(),
                                           nn.Linear(self.projection_dim, self.projection_dim)).to(self.device)
            self.optimizer.add_param_group({'params': self.projector.parameters(), 'lr': self.optimizer.param_groups[0]["lr"]})

        # warm up the mean-teacher framework
        if self.warmup_steps > 0: # not used
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

        # note: if the self.model is never reset, like for continual adaptation,
        # then skipping the state copy would save memory
        self.models = [self.model, self.model_ema, self.projector]
        self.model_states, self.optimizer_state = self.copy_model_and_optimizer()

        if "weight" in self.org_class_relation_form:
            self.class_C_simmat = self.obtain_sim_mat_weight(usage='class Cosine simmat')
            self.class_E_simmat = self.obtain_sim_mat_E_weight(usage='class Euclidean simmat')
        else:
            self.class_C_simmat = self.obtain_sim_mat_proto(usage='class Cosine simmat')
            self.class_E_simmat = self.obtain_sim_mat_E_proto(usage='class Euclidean simmat')

        self.total_steps = 0
        self.seen = []

    @torch.enable_grad()  # ensure grads in possible no grad context for testing
    def warmup(self):
        logger.info(f"Starting warm up...")
        for i in range(self.warmup_steps):
            #linearly increase the learning rate
            for par in self.optimizer.param_groups:
                par["lr"] = self.final_lr * (i+1) / self.warmup_steps

            # sample source batch
            try:
                batch = next(self.src_loader_iter)
            except StopIteration:
                self.src_loader_iter = iter(self.src_loader)
                batch = next(self.src_loader_iter)

            imgs_src, labels_src = batch[0], batch[1]
            imgs_src, labels_src = imgs_src.to(self.device), labels_src.to(self.device).long()

            # forward the test data and optimize the model
            outputs = self.model(imgs_src)
            outputs_ema = self.model_ema(imgs_src)
            loss = self.symmetric_cross_entropy(outputs, outputs_ema).mean(0)
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

            self.model_ema = update_ema_variables(ema_model=self.model_ema, model=self.model, alpha_teacher=self.m_teacher_momentum)

        logger.info(f"Finished warm up...")
        for par in self.optimizer.param_groups:
            par["lr"] = self.final_lr

    def cal_prototype(self, z1, y1):
        start_i = 0
        end_i = self.num_classes
        dim = z1.shape[1]
        current_classes_mean_z1 = torch.zeros((end_i, dim), device=z1.device)
        for i in range(start_i, end_i):
            indices = (y1 == i)
            if any(indices):
                t_z1 = z1[indices]
                mean_z1 = torch.mean(t_z1, dim=0)
                current_classes_mean_z1[i] = mean_z1

        nonZeroRows = torch.abs(current_classes_mean_z1).sum(dim=1) > 0
        nonZero_prototype_z1 = current_classes_mean_z1[nonZeroRows]

        return nonZero_prototype_z1, nonZeroRows

    def topology_preserving_loss(self, s_mean_sub, s_tilde_mean_sub):
        """
        Calculate the topology preserving loss between the original similarity matrix `s`
        and the updated similarity matrix `s_tilde` based on the provided formula.

        :param s: Original similarity matrix (NxN)
        :param s_tilde: Updated similarity matrix (NxN)
        :return: Topology preserving loss value
        """
        # Mean subtraction from the similarity matrices
        # s_mean_sub = s
        # s_tilde_mean_sub = s_tilde

        # Numerator: Covariance between the original and updated similarities
        numerator = (s_mean_sub * s_tilde_mean_sub).sum()

        # Denominator: Standard deviation of the original and updated similarities
        denominator = torch.sqrt(((s_mean_sub ** 2).sum()) * ((s_tilde_mean_sub ** 2).sum()))

        # Topology preserving loss
        loss_tpl = -numerator / denominator

        return loss_tpl

    @torch.enable_grad()  # ensure grads in possible no grad context for testing
    def forward_and_adapt(self, x):

        self.total_steps += 1

        imgs_test = x[0]

        self.optimizer.zero_grad()

        # forward original test data
        features_test = self.feature_extractor(imgs_test)
        outputs_test = self.classifier(features_test)

        # forward augmented test data
        features_aug_test = self.feature_extractor(self.tta_transform((imgs_test)))
        outputs_aug_test = self.classifier(features_aug_test)

        # get the entropy from the stu model
        ent_stu = softmax_entropy(outputs_test)

        # forward original test data through the ema model
        outputs_ema = self.model_ema(imgs_test)
        # ent_tea = softmax_entropy(outputs_ema)
        pres = outputs_test + outputs_ema

        filter_ids = torch.where(ent_stu < self.e_margin)
        self.num_samples_low_ent += filter_ids[0].size(0)

        image_low_ent = imgs_test[filter_ids]
        low_ent_logits = outputs_test[filter_ids]
        low_ent_stu = ent_stu[filter_ids]  # x_buffer, y_buffer
        low_ent_y = pres[filter_ids].argmax(1)

        loss_self_training = (0.5 * self.symmetric_cross_entropy(outputs_test, outputs_ema) + 0.5 * self.symmetric_cross_entropy(outputs_aug_test, outputs_ema)).mean(0)
        loss_trg = self.lambda_ce_trg * loss_self_training

        ori_mem_x = None
        if self.lambda_ce_low_ent > 0:
            ori_mem_x, ori_mem_y, ents, logits = self.buffer.sample(self.buffer_batch_size)
            if ori_mem_x is not None:
                features_mem = self.feature_extractor(ori_mem_x)
                outputs_mem = self.classifier(features_mem)

                loss_ce_mem = F.cross_entropy(outputs_mem, ori_mem_y)
                loss_ce_mem *= self.lambda_ce_low_ent
                loss_ce_mem.backward(retain_graph=True)
            else:
                pass

        ###############################################
        if self.lambda_CRP > 0.0 and ori_mem_x is not None:
            nonZero_prototype_z1, nonZeroRows = self.cal_prototype(features_mem,
                                                                   ori_mem_y)
            if "c" in self.class_relation_type or "C" in self.class_relation_type:
                nonZero_prototype_z1 = F.normalize(nonZero_prototype_z1)
                sim_mat_orig = nonZero_prototype_z1 @ nonZero_prototype_z1.T
                eye_mat = torch.eye(nonZero_prototype_z1.size(0)).to(self.device)
                non_eye_mat = 1 - eye_mat
                sim_mat = eye_mat + non_eye_mat * sim_mat_orig * self.non_diag_alpha
                orig_class_simmat = self.class_C_simmat[nonZeroRows][:, nonZeroRows]
            else:
                sim_mat = torch.cdist(nonZero_prototype_z1, nonZero_prototype_z1)
                orig_class_simmat = self.class_E_simmat[nonZeroRows][:, nonZeroRows]

            if "m" in self.class_relation_loss_type or "M" in self.class_relation_loss_type:
                mse_loss = F.mse_loss(orig_class_simmat, sim_mat, reduction='mean')
                loss_trg += mse_loss * self.lambda_CRP
            if "t" in self.class_relation_loss_type or "T" in self.class_relation_loss_type:
                topology_loss = self.topology_preserving_loss(orig_class_simmat, sim_mat)
                loss_trg += topology_loss * self.lambda_CRP
        ###############################################

        loss_trg.backward()

        if self.lambda_ce_src > 0:
            print("using lambda_ce_src")
            # sample source batch
            try:
                batch = next(self.src_loader_iter)
            except StopIteration:
                self.src_loader_iter = iter(self.src_loader)
                batch = next(self.src_loader_iter)

            # train on labeled source data
            imgs_src, labels_src = batch[0], batch[1]
            features_src = self.feature_extractor(imgs_src.to(self.device))
            outputs_src = self.classifier(features_src)
            loss_ce_src = F.cross_entropy(outputs_src, labels_src.to(self.device).long())
            loss_ce_src *= self.lambda_ce_src
            loss_ce_src.backward()

        self.optimizer.step()

        if self.lambda_ce_low_ent > 0:
            self.buffer.add_reservoir_ent(x=image_low_ent.detach(), y=low_ent_y.detach(), ents=low_ent_stu.detach(),
                                    logits=low_ent_logits.detach())

        self.model_ema = update_ema_variables(ema_model=self.model_ema, model=self.model, alpha_teacher=self.m_teacher_momentum)

        # Stochastic restore
        if self.rst > 0.:
            for nm, m in self.model.named_modules():
                for npp, p in m.named_parameters():
                    if npp in ['weight', 'bias'] and p.requires_grad:
                        mask = (torch.rand(p.shape) < self.rst).float().cuda()
                        with torch.no_grad():
                            p.data = self.model_states[0][f"{nm}.{npp}"] * mask + p * (1.-mask)

        # create and return the ensemble prediction
        return outputs_test + outputs_ema

    @torch.no_grad()
    def forward_sliding_window(self, x):
        """
        Create the prediction for single sample test-time adaptation with a sliding window
        :param x: The buffered data created with a sliding window
        :return: Model predictions
        """
        imgs_test = x[0]
        outputs_test = self.model(imgs_test)
        outputs_ema = self.model_ema(imgs_test)
        return outputs_test + outputs_ema

    def configure_model(self):
        """Configure model"""
        # model.train()
        self.model.eval()  # eval mode to avoid stochastic depth in swin. test-time normalization is still applied
        # disable grad, to (re-)enable only what we update
        self.model.requires_grad_(False)
        # enable all trainable
        for m in self.model.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.requires_grad_(True)
                # force use of batch stats in train and eval modes
                m.track_running_stats = False
                m.running_mean = None
                m.running_var = None
            elif isinstance(m, nn.BatchNorm1d):
                m.train()   # always forcing train mode in bn1d will cause problems for single sample tta
                m.requires_grad_(True)
            else:
                m.requires_grad_(True)

    def collect_params(self):
        """Collect all trainable parameters.
        Walk the model's modules and collect all parameters.
        Return the parameters and their names.
        Note: other choices of parameterization are possible!
        """
        params = []
        names = []
        for nm, m in self.model.named_modules():
            for np, p in m.named_parameters():
                if np in ['weight', 'bias'] and p.requires_grad:
                    params.append(p)
                    names.append(f"{nm}.{np}")
        params = params[:-2]
        names = names[:-2]
        return params, names

    def obtain_sim_mat_weight(self, usage):
        fc_weight = self.classifier.weight.data.detach()
        normalized_fc_weight = F.normalize(fc_weight)
        sim_mat_orig = normalized_fc_weight @ normalized_fc_weight.T
        eye_mat = torch.eye(self.num_classes).to(self.device)
        non_eye_mat = 1 - eye_mat
        sim_mat = (eye_mat + non_eye_mat * sim_mat_orig * self.non_diag_alpha).clone()
        return sim_mat

    def obtain_sim_mat_proto(self, usage):
        fc_weight = self.prototypes_src.squeeze(1).detach()
        normalized_fc_weight = F.normalize(fc_weight)
        sim_mat_orig = normalized_fc_weight @ normalized_fc_weight.T
        eye_mat = torch.eye(self.num_classes).to(self.device)
        non_eye_mat = 1 - eye_mat
        sim_mat = (eye_mat + non_eye_mat * sim_mat_orig * self.non_diag_alpha).clone()
        return sim_mat

    def obtain_sim_mat_E_weight(self, usage):
        fc_weight = self.classifier.weight.data.detach()
        sim_mat = torch.cdist(fc_weight, fc_weight).clone()
        return sim_mat

    def obtain_sim_mat_E_proto(self, usage):
        fc_weight = self.prototypes_src.squeeze(1).detach()
        sim_mat = torch.cdist(fc_weight, fc_weight).clone()
        return sim_mat

@torch.jit.script
def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)