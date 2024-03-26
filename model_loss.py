import math
import torch
import torch.nn as nn
from loguru import logger

from box import bbox_iou
from model import Model
from data_table import DATASET_NUM_DIMS


def create_optimizer(model: nn.Module, name="Adam", lr=0.001, momentum=0.9, decay=1e-5) -> torch.optim.Optimizer:
    # YOLOv5 3-param group optimizer: 0) weights with decay, 1) weights no decay, 2) biases no decay
    g = [[], [], []]  # optimizer parameter groups
    bn = tuple(v for k, v in nn.__dict__.items() if "Norm" in k)  # normalization layers, i.e. BatchNorm2d()
    for v in model.modules():
        for p_name, p in v.named_parameters(recurse=0):
            if p_name == "bias":  # bias (no decay)
                g[2].append(p)
            elif p_name == "weight" and isinstance(v, bn):  # weight (no decay)
                g[1].append(p)
            else:
                g[0].append(p)  # weight (with decay)

    for i in range(3):
        g[i] = [x for x in g[i] if x.requires_grad]

    if name == "Adam":
        optimizer = torch.optim.Adam(g[2], lr=lr, betas=(momentum, 0.999))  # adjust beta1 to momentum
    elif name == "AdamW":
        optimizer = torch.optim.AdamW(g[2], lr=lr, betas=(momentum, 0.999), weight_decay=0.0)
    elif name == "RMSProp":
        optimizer = torch.optim.RMSprop(g[2], lr=lr, momentum=momentum)
    elif name == "SGD":
        optimizer = torch.optim.SGD(g[2], lr=lr, momentum=momentum, nesterov=True)
    else:
        raise NotImplementedError(f"Optimizer {name} not implemented.")

    optimizer.add_param_group({"params": g[0], "weight_decay": decay})  # add g0 with weight_decay
    optimizer.add_param_group({"params": g[1], "weight_decay": 0.0})  # add g1 (BatchNorm2d weights)
    logger.success(f"optimizer: {type(optimizer).__name__}(lr={lr}) with parameter groups " f"{len(g[1])} weight(decay=0.0), {len(g[0])} weight(decay={decay}), {len(g[2])} bias")
    return optimizer


def one_cycle_lambda(y1=0.0, y2=1.0, steps=100):
    # lambda function for sinusoidal ramp from y1 to y2 https://arxiv.org/pdf/1812.01187.pdf
    return lambda x: ((1 - math.cos(x * math.pi / steps)) / 2) * (y2 - y1) + y1


def linear_lambda(lrf, epochs):
    return lambda x: (1 - x / epochs) * (1.0 - lrf) + lrf  # linear


def create_scheduler(optimizer, lrf, final_epoch=300, cos_lr=False):
    if cos_lr:
        lf = one_cycle_lambda(1, lrf, final_epoch)  # cosine 1->hyp['lrf']
    else:
        lf = linear_lambda(lrf, final_epoch)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)  # plot_lr_scheduler(optimizer, scheduler, epochs)
    return scheduler, lf


class EarlyStop:
    # YOLOv5 simple early stopper
    def __init__(self, patience=30):
        self.best_fitness = 0.0  # i.e. mAP
        self.best_epoch = 0
        self.patience = patience or float("inf")  # epochs to wait after fitness stops improving to stop
        self.possible_stop = False  # possible stop may occur next epoch

    def __call__(self, epoch, fitness):
        if fitness >= self.best_fitness:  # >= 0 to allow for early zero-fitness stage of training
            self.best_epoch = epoch
            self.best_fitness = fitness
        delta = epoch - self.best_epoch  # epochs without improvement
        self.possible_stop = delta >= (self.patience - 1)  # possible stop may occur next epoch
        stop = delta >= self.patience  # stop training if patience exceeded
        if stop:
            logger.info(
                f"Stopping training early as no improvement observed in last {self.patience} epochs. "
                f"Best results observed at epoch {self.best_epoch}, best model saved.\n"
                f"To update EarlyStopping(patience={self.patience}) pass a new patience value, "
                f"i.e. `python train.py --patience 300` or use `--patience 0` to disable EarlyStopping."
            )
        return stop


def smooth_BCE(eps=0.1):  # https://github.com/ultralytics/yolov3/issues/238#issuecomment-598028441
    # return positive, negative label smoothing BCE targets
    return 1.0 - 0.5 * eps, 0.5 * eps


class BCEBlurWithLogitsLoss(nn.Module):
    # BCEwithLogitLoss() with reduced missing label effects.
    def __init__(self, alpha=0.05):
        super().__init__()
        self.loss_fcn = nn.BCEWithLogitsLoss(reduction="none")  # must be nn.BCEWithLogitsLoss()
        self.alpha = alpha

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        pred = torch.sigmoid(pred)  # prob from logits
        dx = pred - true  # reduce only missing label effects
        # dx = (pred - true).abs()  # reduce missing label and false label effects
        alpha_factor = 1 - torch.exp((dx - 1) / (self.alpha + 1e-4))
        loss *= alpha_factor
        return loss.mean()


class FocalLoss(nn.Module):
    # Wraps focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super().__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = "none"  # required to apply FL to each element

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        # p_t = torch.exp(-loss)
        # loss *= self.alpha * (1.000001 - p_t) ** self.gamma  # non-zero power for gradient stability

        # TF implementation https://github.com/tensorflow/addons/blob/v0.7.1/tensorflow_addons/losses/focal_loss.py
        pred_prob = torch.sigmoid(pred)  # prob from logits
        p_t = true * pred_prob + (1 - true) * (1 - pred_prob)
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = (1.0 - p_t) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:  # 'none'
            return loss


class QFocalLoss(nn.Module):
    # Wraps Quality focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super().__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = "none"  # required to apply FL to each element

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)

        pred_prob = torch.sigmoid(pred)  # prob from logits
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = torch.abs(true - pred_prob) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:  # 'none'
            return loss


class YOLOv5Loss:
    sort_obj_iou = False

    # Compute losses
    def __init__(self, anchors, anchor_t, na=3, nc=1, nl=3, strides=(8, 16, 32), autobalance=False):
        # Define criteria
        BCEcls = nn.BCEWithLogitsLoss()
        BCEobj = nn.BCEWithLogitsLoss()

        # Class label smoothing https://arxiv.org/pdf/1902.04103.pdf eqn 3
        self.cp, self.cn = smooth_BCE(eps=0.0)  # positive, negative BCE targets

        # Focal loss
        fl_gamma = 0.0  # focal loss gamma
        if fl_gamma > 0:
            BCEcls, BCEobj = FocalLoss(BCEcls, fl_gamma), FocalLoss(BCEobj, fl_gamma)

        self.balance = {3: [4.0, 1.0, 0.4]}.get(nl, [4.0, 1.0, 0.25, 0.06, 0.02])  # P3-P7
        self.ssi = list(strides).index(16) if autobalance else 0  # stride 16 index
        self.BCEcls, self.BCEobj, self.gr, self.autobalance = BCEcls, BCEobj, 1.0, autobalance
        self.na = na  # number of anchors
        self.nc = nc  # number of classes
        self.nl = nl  # number of layers
        self.anchors = anchors
        self.anchor_t = anchor_t

    def get_loss(self, p, tcls, tbox, indices, anchors):  # predictions, targets
        device = p[0].device
        lcls = torch.zeros(1, device=device)  # class loss
        lbox = torch.zeros(1, device=device)  # box loss
        lobj = torch.zeros(1, device=device)  # object loss

        # Losses
        for i, pi in enumerate(p):  # layer index, layer predictions
            b, a, gj, gi = indices[i]  # image, anchor, gridy, gridx
            tobj = torch.zeros(pi.shape[:4], dtype=pi.dtype, device=device)  # target obj

            n = b.shape[0]  # number of targets
            if n:
                # pxy, pwh, _, pcls = pi[b, a, gj, gi].tensor_split((2, 4, 5), dim=1)  # faster, requires torch 1.8.0
                if self.nc > 1:
                    pxy, pwh, _, pcls = pi[b, a, gj, gi].split((2, 2, 1, self.nc, DATASET_NUM_DIMS - 5), 1)  # target-subset of predictions
                else:
                    pxy, pwh, _, _ = pi[b, a, gj, gi].split((2, 2, 1, DATASET_NUM_DIMS - 5), 1)  # target-subset of predictions

                # Regression
                pxy = pxy.sigmoid() * 2 - 0.5
                pwh = (pwh.sigmoid() * 2) ** 2 * anchors[i]
                pbox = torch.cat((pxy, pwh), 1)  # predicted box
                iou = bbox_iou(pbox, tbox[i], CIoU=True).squeeze()  # iou(prediction, target)
                lbox += (1.0 - iou).mean()  # iou loss

                # Objectness
                iou = iou.detach().clamp(0).type(tobj.dtype)
                if self.sort_obj_iou:
                    j = iou.argsort()
                    b, a, gj, gi, iou = b[j], a[j], gj[j], gi[j], iou[j]
                if self.gr < 1:
                    iou = (1.0 - self.gr) + self.gr * iou
                tobj[b, a, gj, gi] = iou  # iou ratio

                # Classification
                if self.nc > 1:  # cls loss (only if multiple classes)
                    t = torch.full_like(pcls, self.cn, device=device)  # targets
                    t[range(n), tcls[i]] = self.cp
                    lcls += self.BCEcls(pcls, t)  # BCE

                # Append targets to text file
                # with open('targets.txt', 'a') as file:
                #     [file.write('%11.5g ' * 4 % tuple(x) + '\n') for x in torch.cat((txy[i], twh[i]), 1)]

            obji = self.BCEobj(pi[..., 4], tobj)
            lobj += obji * self.balance[i]  # obj loss
            if self.autobalance:
                self.balance[i] = self.balance[i] * 0.9999 + 0.0001 / obji.detach().item()

        if self.autobalance:
            self.balance = [x / self.balance[self.ssi] for x in self.balance]
        lbox *= 0.05
        lobj *= 1.0
        lcls *= 0.5
        return lbox + lobj + lcls

    def build_targets(self, p, targets):
        device = p[0].device
        # Build targets for compute_loss(), input targets(image,class,x,y,w,h)
        na, nt = self.na, targets.shape[0]  # number of anchors, targets
        tbox, tcls, tatt, indices, anch = [], [], [], [], []
        gain = torch.ones(DATASET_NUM_DIMS + 2, device=device)  # normalized to gridspace gain
        ai = torch.arange(na, device=device).float().view(na, 1).repeat(1, nt)  # same as .repeat_interleave(nt)
        targets = torch.cat((targets.repeat(na, 1, 1), ai[..., None]), 2)  # append anchor indices

        g = 0.5  # bias
        off = (
            torch.tensor(
                [
                    [0, 0],
                    [1, 0],
                    [0, 1],
                    [-1, 0],
                    [0, -1],  # j,k,l,m
                    # [1, 1], [1, -1], [-1, 1], [-1, -1],  # jk,jm,lk,lm
                ],
                device=device,
            ).float()
            * g  # noqa: W503
        )  # offsets

        for i in range(self.nl):
            anchors, shape = self.anchors[i], p[i].shape
            gain[2:6] = torch.tensor(shape)[[3, 2, 3, 2]]  # xyxy gain

            # Match targets to anchors
            t = targets * gain  # shape(3,n,32)
            if nt:
                # Matches
                r = t[..., 4:6] / anchors[:, None]  # wh ratio
                j = torch.max(r, 1 / r).max(2)[0] < self.anchor_t  # compare
                # j = wh_iou(anchors, t[:, 4:6]) > model.hyp['iou_t']  # iou(3,n)=wh_iou(anchors(3,2), gwh(n,2))
                t = t[j]  # filter

                # Offsets
                gxy = t[:, 2:4]  # grid xy
                gxi = gain[[2, 3]] - gxy  # inverse
                j, k = ((gxy % 1 < g) & (gxy > 1)).T
                l, m = ((gxi % 1 < g) & (gxi > 1)).T
                j = torch.stack((torch.ones_like(j), j, k, l, m))
                t = t.repeat((5, 1, 1))[j]
                offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j]
            else:
                t = targets[0]
                offsets = 0

            # Define
            bc, gxy, gwh, attr, a = t.split([2, 2, 2, DATASET_NUM_DIMS - 5, 1], 1)  # (image, class), grid xy, grid wh, attributes, anchors
            # bc, gxy, gwh, a = t.chunk(4, 1)  # (image, class), grid xy, grid wh, anchors

            a, (b, c) = a.long().view(-1), bc.long().T  # anchors, image, class
            gij = (gxy - offsets).long()
            gi, gj = gij.T  # grid indices

            # Append
            indices.append((b, a, gj.clamp_(0, shape[2] - 1), gi.clamp_(0, shape[3] - 1)))  # image, anchor, grid
            tbox.append(torch.cat((gxy - gij, gwh), 1))  # box
            anch.append(anchors[a])  # anchors
            tcls.append(c)  # class
            tatt.append(attr)

        return tcls, tbox, tatt, indices, anch


class AttributeLoss:
    def __init__(self):
        super().__init__()
        self.logit_loss = QFocalLoss(nn.BCEWithLogitsLoss())
        self.smooth_l1_loss = nn.SmoothL1Loss()

    def get_loss(self, p, tatt, indices):
        device = p[0].device
        llmk = torch.zeros(1, device=device)  # Landmarks loss
        lsta = torch.zeros(1, device=device)  # Status loss
        lqty = torch.zeros(1, device=device)  # Quality loss
        lage = torch.zeros(1, device=device)  # Age loss
        lgdr = torch.zeros(1, device=device)  # Gender loss
        lado = torch.zeros(1, device=device)  # Adornments loss

        for i, pi in enumerate(p):  # layer index, layer predictions
            b, a, gj, gi = indices[i]  # image, anchor, gridy, gridx

            n = b.shape[0]  # number of targets
            if n:
                _, patt = pi[b, a, gj, gi].split((5, DATASET_NUM_DIMS - 5), 1)  # target-subset of predictions
                llmk += self.smooth_l1_loss(torch.sigmoid(patt[:, :10]), tatt[i][:, :10])  # landmarks
                lsta += self.logit_loss(patt[:, 10:12], tatt[i][:, 10:12])  # eye_status
                lsta += self.logit_loss(patt[:, 12:19], tatt[i][:, 12:19])  # occlusions
                lqty += self.logit_loss(patt[:, 19], tatt[i][:, 19])  # blur
                lqty += self.logit_loss(patt[:, 20], tatt[i][:, 20])  # illumination
                lqty += self.logit_loss(patt[:, 21], tatt[i][:, 21])  # completeness
                lage += self.smooth_l1_loss(torch.sigmoid(patt[:, 22]), tatt[i][:, 22])  # age
                lgdr += self.logit_loss(patt[:, 23], tatt[i][:, 23])  # prob_male
                lado += self.logit_loss(patt[:, 24], tatt[i][:, 24])  # wear_glasses
                lado += self.logit_loss(patt[:, 25], tatt[i][:, 25])  # wear_mask

        return llmk, lsta, lqty, lage, lgdr, lado


class YOLOv5FaceQualityLoss:
    def __init__(self, model: Model):
        super().__init__()
        self.yolov5_loss = YOLOv5Loss(model.head.anchors, model.head.anchor_t, model.head.na, 1, model.head.nl, model.head.strides)
        self.attribute_loss = AttributeLoss()

    def __call__(self, input, target):
        tcls, tbox, tetc, indices, anchors = self.yolov5_loss.build_targets(input, target)  # targets
        lbox = self.yolov5_loss.get_loss(input, tcls, tbox, indices, anchors)
        llmk, lsta, lqty, lage, lgdr, lado = self.attribute_loss.get_loss(input, tetc, indices)

        loss = torch.cat((lbox, llmk * 1e1, lsta, lqty, lage * 1e2, lgdr, lado))

        bs = input[0].shape[0]  # batch size
        loss *= bs

        return loss.sum(), loss.detach()
