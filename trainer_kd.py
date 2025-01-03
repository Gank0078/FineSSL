import os
import time
import datetime
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
# from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, SequentialSampler, TensorDataset
from torchvision import transforms

from clip import clip
from model_kd import Model

from utils.meter import AverageMeter
from utils.torchtools import load_checkpoint, save_checkpoint, resume_from_checkpoint

from datasets.data import DATASET_GETTERS
from itertools import cycle

from utils.logger_SSL import Logger
import itertools
from copy import deepcopy

from utils.utils import *
from utils.utils import _ECELoss

from collections import Counter
from collections import OrderedDict
from timm.models.vision_transformer import vit_base_patch16_224, vit_small_patch16_224, vit_tiny_patch16_224, \
    vit_base_patch32_224

best_acc = 0
best_zs_acc = 0
best_acc1 = 0


def load_clip_to_cpu(cfg, teacher=True):
    # backbone_name = cfg.backbone
    if teacher is True:
        backbone_name = cfg.backbone
    else:
        backbone_name = cfg.stu_backbone

    if backbone_name == "IN21K-ViT-B/16":
        model = vit_base_patch16_224(pretrained=True).eval()
        assert cfg.prec in ["fp16", "fp32", "amp"]
        if cfg.prec == "fp16":
            # ViT's default precision is fp32
            model.half()
    elif backbone_name == "IN21K-ViT-S/16":
        model = vit_small_patch16_224(pretrained=True).eval()
        assert cfg.prec in ["fp16", "fp32", "amp"]
        if cfg.prec == "fp16":
            # ViT's default precision is fp32
            model.half()
    elif backbone_name == "IN21K-ViT-T/16":
        model = vit_tiny_patch16_224(pretrained=True).eval()
        assert cfg.prec in ["fp16", "fp32", "amp"]
        if cfg.prec == "fp16":
            # ViT's default precision is fp32
            model.half()
    elif backbone_name == "IN21K-ViT-B/32":
        model = vit_base_patch32_224(pretrained=True).eval()
        assert cfg.prec in ["fp16", "fp32", "amp"]
        if cfg.prec == "fp16":
            # ViT's default precision is fp32
            model.half()
    else:
        url = clip._MODELS[backbone_name]
        model_path = clip._download(url)

        try:
            # loading JIT archive
            model = torch.jit.load(model_path, map_location="cpu").eval()
            state_dict = None

        except RuntimeError:
            state_dict = torch.load(model_path, map_location="cpu")

        model = clip.build_model(state_dict or model.state_dict())

    return model


def compute_adjustment_by_py(py, tro, device):
    adjustments = torch.log(py ** tro + 1e-12)
    adjustments = adjustments.to(device)
    return adjustments


def interleave(x, size):
    s = list(x.shape)
    return x.reshape([-1, size] + s[1:]).transpose(0, 1).reshape([-1] + s[1:])


def de_interleave(x, size):
    s = list(x.shape)
    return x.reshape([size, -1] + s[1:]).transpose(0, 1).reshape([-1] + s[1:])


class Trainer:
    def __init__(self, cfg):

        if not torch.cuda.is_available():
            self.device = torch.device("cpu")
        elif cfg.gpu is None:
            self.device = torch.device("cuda")
        else:
            torch.cuda.set_device(cfg.gpu)
            self.device = torch.device("cuda:{}".format(cfg.gpu))

        # Save as attributes some frequently used variables
        self.start_epoch = self.epoch = 0
        self.num_epochs = cfg.num_epochs
        self.output_dir = cfg.output_dir

        self.cfg = cfg
        self.build_data_loader()
        self.build_model()
        # self.evaluator = Evaluator(cfg, self.cls_num_list)
        self.best_result = -np.inf
        # self._writer = None
        self.th = cfg.th

        class_list = []
        for i in range(cfg.DATA.NUMBER_CLASSES):
            class_list.append(str(i))

        title = 'PEL-SSL-' + cfg.DATA.NAME
        self.logger = Logger(os.path.join(cfg.output_dir, 'logSSL.txt'), title=title)
        self.logger.set_names(['Top1 acc', 'Best Top1 acc', 'epoch'])

        self.logger_alpha = Logger(os.path.join(cfg.output_dir, 'logSSL_alpha.txt'), title=title)
        self.logger_alpha.set_names(['Alpha value', 'epoch'])

        self.logger_half = Logger(os.path.join(cfg.output_dir, 'logSSL_half.txt'), title=title)
        self.logger_half.set_names(['tea ws', 'stu ws', 'ratio', 'le 1.0 count', 'tea le0.1', 'stu le0.1', 'epoch'])

    def build_data_loader(self):
        cfg = self.cfg
        labeled_dataset, unlabeled_dataset, test_dataset = DATASET_GETTERS[cfg.DATA.NAME](
            cfg)

        self.num_classes = cfg.DATA.NUMBER_CLASSES
        self.classnames = labeled_dataset.classes

        self.train_label_loader = DataLoader(labeled_dataset,
                                             batch_size=cfg.DATA.BATCH_SIZE, num_workers=cfg.DATA.NUM_WORKERS,
                                             shuffle=True, drop_last=True, pin_memory=False, persistent_workers=True)

        self.train_unlabel_loader = DataLoader(unlabeled_dataset,
                                               batch_size=cfg.DATA.BATCH_SIZE * self.cfg.DATA.MU_U,
                                               num_workers=cfg.DATA.NUM_WORKERS, shuffle=True,
                                               drop_last=True, pin_memory=False, persistent_workers=True)

        self.test_loader = DataLoader(
            test_dataset,
            sampler=SequentialSampler(test_dataset),
            batch_size=100, pin_memory=False)

    def build_model(self):
        cfg = self.cfg
        # classnames = self.classnames

        # print(f"Loading CLIP (backbone: {cfg.backbone})")
        # clip_model = load_clip_to_cpu(cfg)
        # clip_model.to(self.device)
        clip_model = load_clip_to_cpu(cfg, teacher=True)
        stu_clip_model = load_clip_to_cpu(cfg, teacher=False)
        clip_model.to(self.device)
        stu_clip_model.to(self.device)

        print(cfg.prec)

        assert cfg.prec in ["fp16", "fp32", "amp"]
        if cfg.prec == "fp32" or cfg.prec == "amp":
            # CLIP's default precision is fp16
            clip_model.float()
            stu_clip_model.float()

        if cfg.template is not None:
            temp = cfg.template
        else:
            temp = "a photo of a {}."
        print(temp)
        print(self.classnames)
        prompts = [temp.format(c.replace("_", " ")) for c in self.classnames]
        # prompts = [c for c in self.classnames]
        prompts = torch.cat([clip.tokenize(p) for p in prompts])
        prompts = prompts.to(self.device)

        # with torch.no_grad():
        #     text_features = clip_model.encode_text(prompts)
        #     text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        #
        # self.text_features = text_features
        self.text_features = None
        print("Building model")
        # self.model = Model(cfg, clip_model, self.text_features)
        self.model = Model(cfg, clip_model, self.text_features, teacher=True)
        self.stu_model = Model(cfg, stu_clip_model, self.text_features, teacher=False)

        self.tuner = self.stu_model.tuner
        self.clip_model = clip_model
        # self.dtype = clip_model.dtype
        # self.tuner = self.model.tuner
        # self.clip_model = clip_model
        # # self.dtype = clip_model.dtype

        print("Turning off gradients in the model")
        for name, param in self.model.named_parameters():
            param.requires_grad_(False)

        for name, param in self.tuner.named_parameters():
            param.requires_grad_(True)

        total_params = sum(p.numel() for p in self.model.parameters())
        print(f'Total params: {total_params}')
        tuned_params = sum(p.numel() for p in self.tuner.parameters())
        print(f'Tuned params: {tuned_params}')
        head_params = sum(p.numel() for p in self.tuner.head.parameters())
        tuned_params_without_head = tuned_params - head_params
        print(f'Tuned params (w/o head): {tuned_params_without_head}')

        self.optim = torch.optim.SGD(self.tuner.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay,
                                     momentum=cfg.momentum)
        self.sched = torch.optim.lr_scheduler.CosineAnnealingLR(self.optim, float(cfg.num_epochs))
        self.scaler = GradScaler() if cfg.prec == "amp" else None

        device_count = torch.cuda.device_count()
        if device_count > 1 and cfg.gpu is None:
            print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
            self.model = nn.DataParallel(self.model)
            self.stu_model = nn.DataParallel(self.stu_model)
        self.model.to(self.device)
        self.stu_model.to(self.device)

    def train(self):
        global best_acc
        global best_zs_acc
        global best_acc1

        if self.cfg.resume:
            directory = self.cfg.resume
            # self.start_epoch, sd = self.resume_model_if_exist(directory)
            _ = self.resume_model_if_exist(directory)
            # sd = OrderedDict((key.replace('+', '.'), value) for key, value in sd.items())
            # msg = self.model.image_encoder.load_state_dict(sd, strict=False)
            self.start_epoch = 0

        # self.tuner.head1.weight.data = self.tuner.head.weight.data.clone()
        # Initialize summary writer
        writer_dir = os.path.join(self.output_dir, "tensorboard")
        os.makedirs(writer_dir, exist_ok=True)
        print(f"Initialize tensorboard (log_dir={writer_dir})")
        # self._writer = SummaryWriter(log_dir=writer_dir)

        self.w_con = self.cfg.w_con
        ulab_len = len(self.train_unlabel_loader.dataset)

        self.alpha = self.cfg.alpha
        self.smoothing = self.cfg.smoothing
        self.th_min = self.cfg.th_min

        alpha_m = 0.0

        self.betabase = torch.ones((self.num_classes, self.num_classes)).to(self.device)
        self.betabase[torch.arange(self.num_classes), torch.arange(self.num_classes)] = 0.0

        ws_list = []
        alpha = 0.5

        self.time_start = time.time()

        for self.epoch in range(self.start_epoch, self.num_epochs):
            self.tuner.train()

            batch_time = AverageMeter()
            data_time = AverageMeter()
            loss_meter = AverageMeter()
            acc_meter = AverageMeter()
            self.num_batches = self.cfg.total_steps
            label_loader_iter = cycle(self.train_label_loader)
            unlabel_loader_iter = cycle(self.train_unlabel_loader)
            alpha_list = []

            selected_label = torch.ones((ulab_len,), dtype=torch.long, ) * -1
            selected_label = selected_label.to(self.device)
            classwise_acc = torch.zeros((self.num_classes,)).to(self.device)

            end = time.time()

            for self.batch_idx in range(self.cfg.total_steps):
                data_time.update(time.time() - end)

                (inputs_x, targets_x, _) = next(label_loader_iter)
                batch_size = inputs_x.shape[0]
                ((inputs_u_w, inputs_u_s, inputs_u_s1), u_real, uidx) = next(unlabel_loader_iter)

                # uidx = uidx.to(self.device)
                # targets_x = targets_x.to(self.device)
                uidx = uidx.to(device=self.device, non_blocking=True)
                targets_x = targets_x.to(device=self.device, non_blocking=True)
                targets_x = targets_x.to(torch.long)

                pseudo_counter = Counter(selected_label.tolist())

                if max(pseudo_counter.values()) < ulab_len:
                    wo_negative_one = deepcopy(pseudo_counter)
                    if -1 in wo_negative_one.keys():
                        wo_negative_one.pop(-1)
                    for i in range(self.num_classes):
                        classwise_acc[i] = pseudo_counter[i] / max(wo_negative_one.values())

                inputs_x, inputs_u_w, inputs_u_s = (inputs_x.to(device=self.device, non_blocking=True),
                                                    inputs_u_w.to(device=self.device, non_blocking=True), \
                                                    inputs_u_s.to(device=self.device, non_blocking=True))

                inputs = interleave(
                    torch.cat((inputs_x, inputs_u_w, inputs_u_s)), 2 * self.cfg.DATA.MU_U + 1)

                with torch.no_grad():
                    feat = self.model(inputs)
                stu_feat = self.stu_model(inputs)
                feat = de_interleave(feat, 2 * self.cfg.DATA.MU_U + 1)
                stu_feat = de_interleave(stu_feat, 2 * self.cfg.DATA.MU_U + 1)

                with torch.no_grad():
                    tea_output = self.model.tuner.head(feat)
                    # output_x = output[:batch_size]
                    tea_output_u_w, tea_output_u_s = tea_output[batch_size:].chunk(2)

                    tea_output1 = self.model.tuner.head1(feat)
                    tea_output1_u_w, _ = tea_output1[batch_size:].chunk(2)

                stu_output = self.tuner.head(stu_feat)
                stu_output_x = stu_output[:batch_size]
                stu_output_u_w, stu_output_u_s = stu_output[batch_size:].chunk(2)

                stu_output1 = self.tuner.head1(stu_feat.detach())
                stu_output1_x = stu_output1[:batch_size]
                stu_output1_u_w, stu_output1_u_s = stu_output1[batch_size:].chunk(2)

                del feat
                del stu_feat
                del tea_output
                del stu_output

                stu_pseu = F.softmax(stu_output_u_w.detach(), dim=1)
                stu_conf, stu_targets_u = torch.max(stu_pseu, dim=-1)
                stu_ws = F.cross_entropy(stu_output_u_s.detach(), stu_targets_u, reduction='none')
                stu_ws_mean = stu_ws.mean()
                tea_pseu = F.softmax(tea_output_u_w.detach(), dim=1)
                tea_conf, tea_targets_u = torch.max(tea_pseu, dim=-1)
                tea_ws = F.cross_entropy(tea_output_u_s.detach(), tea_targets_u, reduction='none')
                tea_ws_mean = tea_ws.mean()

                ws_ratio = (tea_ws / (stu_ws + 1e-6))
                if self.batch_idx % self.cfg.ep_ws == 0:
                    ws_list.append(tea_ws_mean / stu_ws_mean)
                    ratio = sum(ws_list) / len(ws_list)
                    ws_list = []
                    alpha = alpha * self.cfg.beta + ratio * alpha * (1.0 - self.cfg.beta)
                    # alpha = min(alpha, 1.0 - self.cfg.extval)
                    # alpha = max(alpha, self.cfg.extval)
                    alpha = max(alpha, self.cfg.extval)
                    alpha = min(alpha, 1.0)
                    alpha_list.append(alpha)
                else:
                    ws_list.append(tea_ws_mean / stu_ws_mean)

                # if self.epoch >= self.cfg.kd_epoch:
                alpha_m = self.alpha * alpha

                # mar_diff = (classwise_acc.max() - classwise_acc.min()) * self.alpha
                mar_diff = (classwise_acc.max() - classwise_acc.min()) * alpha_m
                alpha_margin = (((1.0 - classwise_acc) / (classwise_acc + 1.0)) * mar_diff).unsqueeze(1)
                stu_output_x = stu_output_x + alpha_margin[targets_x] * self.betabase[targets_x]
                Lx = F.cross_entropy(stu_output_x, targets_x, reduction='mean')
                Lx1 = F.cross_entropy(stu_output1_x, targets_x, reduction='mean')

                p_tea = F.softmax(tea_output_u_w.detach() / self.cfg.Tr, dim=1)
                logp_stu = F.log_softmax(stu_output_u_w / self.cfg.Tr, dim=1)
                Lu_kd = (torch.sum(-p_tea * logp_stu, dim=1)).mean()

                logp_stu1 = F.log_softmax(stu_output1_u_w / self.cfg.Tr, dim=1)
                p_tea1 = F.softmax(tea_output1_u_w.detach() / self.cfg.Tr, dim=1)
                Lu_kd1 = (torch.sum(-p_tea1 * logp_stu1, dim=1)).mean()

                pseu = torch.softmax(stu_output_u_w.detach(), dim=-1)
                conf, targets_u = torch.max(pseu, dim=-1)
                mask = conf.ge(self.th)

                pseu1 = torch.softmax(stu_output1_u_w.detach(), dim=-1)
                conf1, targets1_u = torch.max(pseu1, dim=-1)
                confw = conf1
                mask1 = conf1.ge(self.th_min)
                stu_output_u_s = stu_output_u_s + alpha_margin[targets_u] * self.betabase[targets_u]
                if torch.sum(mask1) > 0:
                    Lu = (F.cross_entropy(stu_output_u_s, targets_u,
                                          reduction='none') * confw * mask1).mean()
                else:
                    Lu = 0

                onehotu = targets_u.reshape(-1, 1)
                onehotu = torch.zeros_like(stu_output1_u_s).scatter(1, onehotu, 1)
                onehotu = onehotu * (1 - self.smoothing) + (1 - onehotu) * self.smoothing / (self.num_classes - 1)
                log_predu = F.log_softmax(stu_output1_u_s, dim=-1)

                if torch.sum(mask) > 0:
                    Lu1 = torch.sum(-onehotu * log_predu, dim=1)[mask].mean()
                else:
                    Lu1 = 0

                select = conf.ge(0.7).long()
                if uidx[select == 1].nelement() != 0:
                    selected_label[uidx[select == 1]] = targets_u[select == 1]

                loss = Lx + Lx1 + ((Lu_kd + Lu_kd1) * (1 - alpha) + (Lu * self.cfg.s_con + Lu1) * alpha) * self.cfg.w_con

                self.optim.zero_grad()
                loss.backward()
                self.optim.step()

                current_logit_scale = self.tuner.head.logit_scale.item()
                current_lr = self.optim.param_groups[0]["lr"]
                loss_meter.update(loss.item())

                batch_time.update(time.time() - end)

                meet_freq = (self.batch_idx + 1) % self.cfg.print_freq == 0
                only_few_batches = self.num_batches < self.cfg.print_freq
                if meet_freq or only_few_batches:
                    nb_remain = 0
                    nb_remain += self.num_batches - self.batch_idx - 1
                    nb_remain += (
                                         self.num_epochs - self.epoch - 1
                                 ) * self.num_batches
                    eta_seconds = batch_time.avg * nb_remain
                    eta = str(datetime.timedelta(seconds=int(eta_seconds)))

                    info = []
                    info += [f"epoch [{self.epoch + 1}/{self.num_epochs}]"]
                    info += [f"batch [{self.batch_idx + 1}/{self.num_batches}]"]
                    info += [f"loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})"]
                    info += [f"acc {acc_meter.val:.4f} ({acc_meter.avg:.4f})"]
                    info += [f"s {current_logit_scale:.4f}"]
                    info += [f"lr {current_lr:.4e}"]
                    info += [f"eta {eta}"]
                    print(" ".join(info))

                n_iter = self.epoch * self.num_batches + self.batch_idx
                # self._writer.add_scalar("train/loss", loss_meter.avg, n_iter)
                # self._writer.add_scalar("train/acc", acc_meter.avg, n_iter)
                # self._writer.add_scalar("train/lr", current_lr, n_iter)

                if (self.batch_idx + 1) == self.num_batches:
                    self.sched.step()

                end = time.time()

            last_epoch = (self.epoch + 1) == self.num_epochs
            meet_checkpoint_freq = (
                (self.epoch + 1) % self.cfg.checkpoint_freq == 0
                if self.cfg.checkpoint_freq > 0 else False
            )

            if meet_checkpoint_freq or last_epoch:
                self.save_model(self.epoch, self.output_dir)

            # torch.cuda.empty_cache()

            acc_now = self.test()
            best_acc = max(best_acc, acc_now)
            self.logger.append([acc_now, best_acc, self.epoch + 1])
            alpha_mean = sum(alpha_list) / len(alpha_list)
            self.logger_alpha.append([alpha_mean, self.epoch + 1])

        print("Finish training")

        print("Deploy the last-epoch model for testing")

        # Show elapsed time
        elapsed = round(time.time() - self.time_start)
        elapsed = str(datetime.timedelta(seconds=elapsed))
        print(f"Elapsed: {elapsed}")

        # Close writer
        # self._writer.close()

        self.logger.close()
        self.logger_alpha.close()
        self.logger_half.close()

    @torch.no_grad()
    def test(self):
        self.tuner.eval()
        print(f"Evaluate on the test set")
        preds = np.array([])
        targets = np.array([])
        for batch in tqdm(self.test_loader, ascii=True):
            image = batch[0]
            label = batch[1]
            image = image.to(self.device, non_blocking=True)
            label = label.to(self.device, non_blocking=True)
            # feat = self.model(image)
            feat = self.stu_model(image)  # using student model for prediction
            output = self.tuner.head(feat)
            del feat

            prob = F.softmax(output, dim=1)
            conf, pred = prob.max(1)
            preds = np.append(preds, pred.cpu().numpy())
            targets = np.append(targets, label.cpu().numpy())

        targets = targets.astype(int)
        preds = preds.astype(int)
        acc = sum(targets == preds) / len(targets)

        return acc

    def save_model(self, epoch, directory, is_best=False, val_result=None, model_name=""):
        tuner_dict = self.tuner.state_dict()
        optim_dict = self.optim.state_dict()
        sched_dict = self.sched.state_dict()
        save_checkpoint(
            {
                "state_dict": tuner_dict,
                "epoch": epoch + 1,
                "optimizer": optim_dict,
                "scheduler": sched_dict,
                "val_result": val_result
            },
            os.path.join(directory + f'/epoch_{epoch+1}', "tuner"),
            is_best=is_best,
            model_name=model_name,
        )

    def resume_model_if_exist(self, directory):
        file_missing = False

        path = os.path.join(directory, "tuner")
        if not os.path.exists(path):
            print("No checkpoint found, train from scratch")
            return 0

        print(f"Found checkpoint at {directory} (will resume training)")

        path = os.path.join(directory, "tuner")
        # start_epoch, sd = resume_from_checkpoint(
        #     path, self.tuner, self.optim, self.sched
        # )

        start_epoch, sd = resume_from_checkpoint(
            path, self.model.tuner, self.optim, self.sched
        )

        return start_epoch, sd

    def load_model(self, directory, epoch=None):
        if not directory:
            print("Note that load_model() is skipped as no pretrained model is given")
            return

        # By default, the best model is loaded
        model_file = "model-best.pth.tar"

        if epoch is not None:
            model_file = "model.pth.tar-" + str(epoch)

        model_path = os.path.join(directory, "tuner", model_file)

        if not os.path.exists(model_path):
            raise FileNotFoundError('Model not found at "{}"'.format(model_path))

        checkpoint = load_checkpoint(model_path, self.device)
        state_dict = checkpoint["state_dict"]
        epoch = checkpoint["epoch"]

        # Ignore fixed token vectors
        if "token_prefix" in state_dict:
            del state_dict["token_prefix"]

        if "token_suffix" in state_dict:
            del state_dict["token_suffix"]

        print("Loading weights to {} " 'from "{}" (epoch = {})'.format("tuner", model_path, epoch))
        # set strict=False
        self.tuner.load_state_dict(state_dict, strict=False)
