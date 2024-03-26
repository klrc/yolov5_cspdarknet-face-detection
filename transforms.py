# YOLOv5 🚀 by Ultralytics, AGPL-3.0 license
"""
Image augmentation functions
"""

import math
import random
import time
import cv2
import numpy as np
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from loguru import logger
from PIL import Image, ImageDraw, ImageFont

from box import box_candidates, xywhn2xyxy, bbox_ioa, xyn2xy
from general import check_version, colorstr
from data_table import IDX_LANDMARKS, IDX_EYE_STATUS, IDX_OCCLUSIONS, IDX_BLUR, IDX_ILLUMINATION, IDX_COMPLETENESS

# Use this for issue: "'locale' codec can't encoder character ..."
# import locale
# locale.setlocale(locale.LC_CTYPE, 'chinese')

IMAGENET_MEAN = 0.485, 0.456, 0.406  # RGB mean
IMAGENET_STD = 0.229, 0.224, 0.225  # RGB standard deviation


def luma_from_bgr(image):
    bgr_avgs = np.average(np.average(image, axis=0), axis=0)  # find avg of each channel
    y_coeffs = np.array([0.114, 0.587, 0.299])  # BGR
    luma_avg = np.dot(y_coeffs, bgr_avgs)
    return luma_avg


class AlbumentationsPreset:
    # YOLOv5 Albumentations class (optional, only used if package is installed)
    def __init__(self, random_illumination=0.0, random_blur=0.0, random_noise=0.0, random_compression=0.0):
        self.transform = None
        prefix = colorstr("albumentations: ")
        try:
            import albumentations as A

            check_version(A.__version__, "1.0.3", hard=True)  # version requirement

            T_NORMAL = [
                A.ToGray(p=0.01),
                A.CLAHE(p=0.01),
                A.OneOf(
                    [
                        A.GaussNoise(),
                        A.ISONoise(),
                        A.Sharpen(),
                    ],
                    p=random_noise,
                ),
                A.OneOf(
                    [
                        A.ImageCompression(quality_lower=25),
                        A.ImageCompression(quality_lower=50),
                        A.ImageCompression(quality_lower=75),
                    ],
                    p=random_compression,
                ),
            ]
            T_ILLUMINATION = [
                A.OneOf(
                    [
                        A.RandomBrightness(),
                    ],
                    p=1,
                ),
            ]
            T_BLUR = [
                A.OneOf(
                    [
                        A.MotionBlur(blur_limit=21),
                    ],
                    p=1,
                ),
            ]

            self.transform_normal = A.Compose(T_NORMAL)
            self.transform_blur = A.Compose(T_BLUR)
            self.transform_illumination = A.Compose(T_ILLUMINATION)
            self.p_blur = random_blur
            self.p_illumination = random_illumination

            logger.info(prefix + ", ".join(f"{x}".replace("always_apply=False, ", "") for x in T_NORMAL if x.p))
            logger.info(prefix + ", ".join(f"{x}".replace("always_apply=False, ", "") for x in T_BLUR if x.p))
            logger.info(prefix + ", ".join(f"{x}".replace("always_apply=False, ", "") for x in T_ILLUMINATION if x.p))
        except ImportError:  # package not installed, skip
            pass
        except Exception as e:
            logger.info(f"{prefix}{e}")

    def __call__(self, im, labels, p=1.0):
        if random.random() < p:
            # Check blur
            if random.random() < self.p_blur:
                im = self.transform_blur(image=im)["image"]
                labels[:, IDX_BLUR] = 1

            # Check illumination
            if random.random() < self.p_illumination:
                raw_luma = luma_from_bgr(im)
                im = self.transform_illumination(image=im)["image"]
                new_luma = luma_from_bgr(im)
                labels[:, IDX_ILLUMINATION] = np.clip(labels[:, IDX_ILLUMINATION] / raw_luma * new_luma, 0, 1)

            # Common transform
            im = self.transform_normal(image=im)["image"]

        return im, labels


def padding(frame: torch.Tensor, pad_h: int, pad_w: int):
    n, c, h, w = frame.shape
    exp_h = h + pad_h * 2
    exp_w = w + pad_w * 2
    background = torch.zeros(n, c, exp_h, exp_w, device=frame.device)
    pad_h = (exp_h - h) // 2
    pad_w = (exp_w - w) // 2
    background[:, :, pad_h : pad_h + h, pad_w : pad_w + w] = frame  # noqa:E203
    return background


def letterbox_padding(frame: torch.Tensor, gs=32):
    n, c, h, w = frame.shape
    if w % gs == 0 and h % gs == 0:
        return frame
    exp_h = math.ceil(h / gs) * gs
    exp_w = math.ceil(w / gs) * gs
    background = torch.zeros(n, c, exp_h, exp_w, device=frame.device)
    pad_h = (exp_h - h) // 2
    pad_w = (exp_w - w) // 2
    background[:, :, pad_h : pad_h + h, pad_w : pad_w + w] = frame  # noqa:E203
    return background


def maxbound_resize(image: np.ndarray, inf_size: int):
    height, width, _ = image.shape
    ratio = inf_size / max(height, width)
    image = cv2.resize(image, (int(ratio * width), int(ratio * height)))
    return image


def normalize(x, mean=IMAGENET_MEAN, std=IMAGENET_STD, inplace=False):
    # Denormalize RGB images x per ImageNet stats in BCHW format, i.e. = (x - mean) / std
    return TF.normalize(x, mean, std, inplace=inplace)


def denormalize(x, mean=IMAGENET_MEAN, std=IMAGENET_STD):
    # Denormalize RGB images x per ImageNet stats in BCHW format, i.e. = x * std + mean
    for i in range(3):
        x[:, i] = x[:, i] * std[i] + mean[i]
    return x


def augment_hsv(im, hgain=0.5, sgain=0.5, vgain=0.5):
    # HSV color-space augmentation
    if hgain or sgain or vgain:
        r = np.random.uniform(-1, 1, 3) * [hgain, sgain, vgain] + 1  # random gains
        hue, sat, val = cv2.split(cv2.cvtColor(im, cv2.COLOR_BGR2HSV))
        dtype = im.dtype  # uint8

        x = np.arange(0, 256, dtype=r.dtype)
        lut_hue = ((x * r[0]) % 180).astype(dtype)
        lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
        lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

        im_hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
        cv2.cvtColor(im_hsv, cv2.COLOR_HSV2BGR, dst=im)  # no return needed


def hist_equalize(im, clahe=True, bgr=False):
    # Equalize histogram on BGR image 'im' with im.shape(n,m,3) and range 0-255
    yuv = cv2.cvtColor(im, cv2.COLOR_BGR2YUV if bgr else cv2.COLOR_RGB2YUV)
    if clahe:
        c = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        yuv[:, :, 0] = c.apply(yuv[:, :, 0])
    else:
        yuv[:, :, 0] = cv2.equalizeHist(yuv[:, :, 0])  # equalize Y channel histogram
    return cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR if bgr else cv2.COLOR_YUV2RGB)  # convert YUV image to RGB


def replicate(im, labels):
    # Replicate labels
    h, w = im.shape[:2]
    boxes = labels[:, 1:].astype(int)
    x1, y1, x2, y2 = boxes.T
    s = ((x2 - x1) + (y2 - y1)) / 2  # side length (pixels)
    for i in s.argsort()[: round(s.size * 0.5)]:  # smallest indices
        x1b, y1b, x2b, y2b = boxes[i]
        bh, bw = y2b - y1b, x2b - x1b
        yc, xc = int(random.uniform(0, h - bh)), int(random.uniform(0, w - bw))  # offset x, y
        x1a, y1a, x2a, y2a = [xc, yc, xc + bw, yc + bh]
        im[y1a:y2a, x1a:x2a] = im[y1b:y2b, x1b:x2b]  # im4[ymin:ymax, xmin:xmax]
        labels = np.append(labels, [[labels[i, 0], x1a, y1a, x2a, y2a]], axis=0)

    return im, labels


def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return im, ratio, (dw, dh)


def random_perspective(im, targets=(), degrees=10, translate=0.1, scale=0.1, shear=10, perspective=0.0, border=(0, 0)):
    # torchvision.transforms.RandomAffine(degrees=(-10, 10), translate=(0.1, 0.1), scale=(0.9, 1.1), shear=(-10, 10))
    # targets = [cls, xyxy]
    targets: np.ndarray

    height = im.shape[0] + border[0] * 2  # shape(h,w,c)
    width = im.shape[1] + border[1] * 2

    # Center
    C = np.eye(3)
    C[0, 2] = -im.shape[1] / 2  # x translation (pixels)
    C[1, 2] = -im.shape[0] / 2  # y translation (pixels)

    # Perspective
    P = np.eye(3)
    P[2, 0] = random.uniform(-perspective, perspective)  # x perspective (about y)
    P[2, 1] = random.uniform(-perspective, perspective)  # y perspective (about x)

    # Rotation and Scale
    R = np.eye(3)
    a = random.uniform(-degrees, degrees)
    # a += random.choice([-180, -90, 0, 90])  # add 90deg rotations to small rotations
    s = random.uniform(1 - scale, 1 + scale)
    # s = 2 ** random.uniform(-scale, scale)
    R[:2] = cv2.getRotationMatrix2D(angle=a, center=(0, 0), scale=s)

    # Shear
    S = np.eye(3)
    S[0, 1] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # x shear (deg)
    S[1, 0] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # y shear (deg)

    # Translation
    T = np.eye(3)
    T[0, 2] = random.uniform(0.5 - translate, 0.5 + translate) * width  # x translation (pixels)
    T[1, 2] = random.uniform(0.5 - translate, 0.5 + translate) * height  # y translation (pixels)

    # Combined rotation matrix
    M = T @ S @ R @ P @ C  # order of operations (right to left) is IMPORTANT
    if (border[0] != 0) or (border[1] != 0) or (M != np.eye(3)).any():  # image changed
        if perspective:
            im = cv2.warpPerspective(im, M, dsize=(width, height), borderValue=(114, 114, 114))
        else:  # affine
            im = cv2.warpAffine(im, M[:2], dsize=(width, height), borderValue=(114, 114, 114))

    # Visualize
    # import matplotlib.pyplot as plt
    # ax = plt.subplots(1, 2, figsize=(12, 6))[1].ravel()
    # ax[0].imshow(im[:, :, ::-1])  # base
    # ax[1].imshow(im2[:, :, ::-1])  # warped

    # Transform label coordinates
    n = len(targets)
    if n:
        new = np.zeros((n, 4))

        xy = np.ones((n * 5, 3))
        xy[:, :2] = targets[:, IDX_LANDMARKS : IDX_LANDMARKS + 10].reshape(-1, 2)
        xy = xy @ M.T  # transform
        xy = xy[:, :2] / xy[:, 2:3] if perspective else xy[:, :2]  # perspective rescale or affine
        xy[:, ::2] = np.clip(xy[:, ::2], 0, width)  # clip x
        xy[:, 1::2] = np.clip(xy[:, 1::2], 0, height)  # clip y
        targets[:, IDX_LANDMARKS : IDX_LANDMARKS + 10] = xy.reshape(-1, 10)

        # use_segments = any(x.any() for x in segments) and len(segments) == n
        # if use_segments:  # warp segments
        #     segments = resample_segments(segments)  # upsample
        #     for i, segment in enumerate(segments):
        #         xy = np.ones((len(segment), 3))
        #         xy[:, :2] = segment
        #         xy = xy @ M.T  # transform
        #         xy = xy[:, :2] / xy[:, 2:3] if perspective else xy[:, :2]  # perspective rescale or affine

        #         # clip
        #         new[i] = segment2box(xy, width, height)

        # else:  # warp boxes
        xy = np.ones((n * 4, 3))
        xy[:, :2] = targets[:, [1, 2, 3, 4, 1, 4, 3, 2]].reshape(n * 4, 2)  # x1y1, x2y2, x1y2, x2y1
        xy = xy @ M.T  # transform
        xy = (xy[:, :2] / xy[:, 2:3] if perspective else xy[:, :2]).reshape(n, 8)  # perspective rescale or affine

        # create new boxes
        x = xy[:, [0, 2, 4, 6]]
        y = xy[:, [1, 3, 5, 7]]
        new = np.concatenate((x.min(1), y.min(1), x.max(1), y.max(1))).reshape(4, n).T

        # clip & update completeness
        raw_area = (new[:, 3] - new[:, 1]) * (new[:, 2] - new[:, 0]) + 1e-7
        new[:, [0, 2]] = new[:, [0, 2]].clip(0, width)
        new[:, [1, 3]] = new[:, [1, 3]].clip(0, height)
        new_area = (new[:, 3] - new[:, 1]) * (new[:, 2] - new[:, 0])
        completeness_loss = np.nan_to_num(new_area / raw_area, nan=0, posinf=0, neginf=0)
        targets[:, IDX_COMPLETENESS] *= completeness_loss

        # filter candidates
        # i = box_candidates(box1=targets[:, 1:5].T * s, box2=new.T, area_thr=0.01 if use_segments else 0.10)
        i = box_candidates(box1=targets[:, 1:5].T * s, box2=new.T, area_thr=0.10)
        targets = targets[i]
        targets[:, 1:5] = new[i]

    return im, targets


def copy_paste(im, labels, segments, p=0.5):
    raise NotImplementedError
    # Implement Copy-Paste augmentation https://arxiv.org/abs/2012.07177, labels as nx5 np.array(cls, xyxy)
    n = len(segments)
    if p and n:
        h, w, c = im.shape  # height, width, channels
        im_new = np.zeros(im.shape, np.uint8)
        for j in random.sample(range(n), k=round(p * n)):
            l, s = labels[j], segments[j]
            box = w - l[3], l[2], w - l[1], l[4]
            ioa = bbox_ioa(box, labels[:, 1:5])  # intersection over area
            if (ioa < 0.30).all():  # allow 30% obscuration of existing labels
                labels = np.concatenate((labels, [[l[0], *box]]), 0)
                segments.append(np.concatenate((w - s[:, 0:1], s[:, 1:2]), 1))
                cv2.drawContours(im_new, [segments[j].astype(np.int32)], -1, (1, 1, 1), cv2.FILLED)

        result = cv2.flip(im, 1)  # augment segments (flip left-right)
        i = cv2.flip(im_new, 1).astype(bool)
        im[i] = result[i]  # cv2.imwrite('debug.jpg', im)  # debug

    return im, labels, segments


def cutout(im, labels, p=0.5):
    # Applies image cutout augmentation https://arxiv.org/abs/1708.04552
    if random.random() < p:
        h, w = im.shape[:2]
        scales = [0.5] * 1 + [0.25] * 2 + [0.125] * 4 + [0.0625] * 8 + [0.03125] * 16  # image size fraction
        for s in scales:
            mask_h = random.randint(1, int(h * s))  # create random masks
            mask_w = random.randint(1, int(w * s))

            # box
            xmin = max(0, random.randint(0, w) - mask_w // 2)
            ymin = max(0, random.randint(0, h) - mask_h // 2)
            xmax = min(w, xmin + mask_w)
            ymax = min(h, ymin + mask_h)

            # apply random color mask
            im[ymin:ymax, xmin:xmax] = [random.randint(64, 191) for _ in range(3)]

            # return unobscured labels
            if len(labels) and s > 0.03:
                box = np.array([xmin, ymin, xmax, ymax], dtype=np.float32)
                ioa = bbox_ioa(box, xywhn2xyxy(labels[:, 1:5], w, h))  # intersection over area
                labels = labels[ioa < 0.60]  # remove >60% obscured labels

    return labels


def mixup(im, labels, im2, labels2):
    # Applies MixUp augmentation https://arxiv.org/pdf/1710.09412.pdf
    r = np.random.beta(32.0, 32.0)  # mixup ratio, alpha=beta=32.0
    im = (im * r + im2 * (1 - r)).astype(np.uint8)
    labels = np.concatenate((labels, labels2), 0)
    return im, labels


def mosaic4(im_list, labels_list, img_size, mosaic_border):
    # YOLOv5 4-mosaic loader. Loads 1 image + 3 random images into a 4-image mosaic
    labels4 = []
    s = img_size
    yc, xc = (int(random.uniform(-x, 2 * s + x)) for x in mosaic_border)  # mosaic center x, y

    for i, (img, labels) in enumerate(zip(im_list, labels_list)):
        h, w, _ = img.shape
        # place img in img4
        if i == 0:  # top left
            img4 = np.full((s * 2, s * 2, img.shape[2]), 114, dtype=np.uint8)  # base image with 4 tiles
            x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc  # xmin, ymin, xmax, ymax (large image)
            x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h  # xmin, ymin, xmax, ymax (small image)
        elif i == 1:  # top right
            x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s * 2), yc
            x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
        elif i == 2:  # bottom left
            x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(s * 2, yc + h)
            x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(y2a - y1a, h)
        elif i == 3:  # bottom right
            x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s * 2), min(s * 2, yc + h)
            x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)

        img4[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]  # img4[ymin:ymax, xmin:xmax]
        padw = x1a - x1b
        padh = y1a - y1b

        # Labels
        if labels.size:
            labels[:, 1:] = xywhn2xyxy(labels[:, 1:], w, h, padw, padh)  # normalized xywh to pixel xyxy format
            for il in range(IDX_LANDMARKS, IDX_LANDMARKS + 10, 2):
                labels[:, il : il + 2] = xyn2xy(labels[:, il : il + 2], w, h, padw, padh)
        labels4.append(labels)

    # Concat/Clip labels
    labels4 = np.concatenate(labels4, 0)
    labels4[:, IDX_LANDMARKS : IDX_LANDMARKS + 10] = np.clip(labels4[:, IDX_LANDMARKS : IDX_LANDMARKS + 10], 0, 2 * s)

    # Update completeness
    raw_area = (labels4[:, 3] - labels4[:, 1]) * (labels4[:, 4] - labels4[:, 2])
    box_is_available = raw_area > 0
    labels4 = labels4[box_is_available]
    raw_area = raw_area[box_is_available]
    labels4[:, 1:5] = np.clip(labels4[:, 1:5], 0, 2 * s)
    new_area = (labels4[:, 3] - labels4[:, 1]) * (labels4[:, 4] - labels4[:, 2])
    completeness_loss = new_area / raw_area
    labels4[:, IDX_COMPLETENESS] *= completeness_loss
    return img4, labels4


def fake_osd(image, ttf_path):
    # Add OSD to image
    # Use simsum.ttc to write Chinese.
    H, W, _ = image.shape
    x, y = random.random() * W, random.random() * H
    color = random.choice([(0, 0, 0, 0), (255, 255, 255, 0)])
    size = random.randint(14, 32)
    font = ImageFont.truetype(ttf_path, size)
    img_pil = Image.fromarray(image)
    draw = ImageDraw.Draw(img_pil)
    # Random timestamp
    start = 189273600  # 1976-01-01 00：00：00
    end = 5133951999  # 1990-12-31 23：59：59
    t = random.randint(start, end)
    date_touple = time.localtime(t)
    weekday = random.choice("一二三四五六日")
    text_content = time.strftime(f"%Y年%m月%d日 星期{weekday} %H:%M:%S", date_touple)
    draw.text((x, y), text_content, font=font, fill=color)
    image = np.array(img_pil)
    return image


def classify_albumentations(
    augment=True,
    size=224,
    scale=(0.08, 1.0),
    ratio=(0.75, 1.0 / 0.75),  # 0.75, 1.33
    hflip=0.5,
    vflip=0.0,
    jitter=0.4,
    mean=IMAGENET_MEAN,
    std=IMAGENET_STD,
    auto_aug=False,
):
    # YOLOv5 classification Albumentations (optional, only used if package is installed)
    prefix = colorstr("albumentations: ")
    try:
        import albumentations as A
        from albumentations.pytorch import ToTensorV2

        check_version(A.__version__, "1.0.3", hard=True)  # version requirement
        if augment:  # Resize and crop
            T = [A.RandomResizedCrop(height=size, width=size, scale=scale, ratio=ratio)]
            if auto_aug:
                # TODO: implement AugMix, AutoAug & RandAug in albumentation
                logger.info(f"{prefix}auto augmentations are currently not supported")
            else:
                if hflip > 0:
                    T += [A.HorizontalFlip(p=hflip)]
                if vflip > 0:
                    T += [A.VerticalFlip(p=vflip)]
                if jitter > 0:
                    color_jitter = (float(jitter),) * 3  # repeat value for brightness, contrast, satuaration, 0 hue
                    T += [A.ColorJitter(*color_jitter, 0)]
        else:  # Use fixed crop for eval set (reproducibility)
            T = [A.SmallestMaxSize(max_size=size), A.CenterCrop(height=size, width=size)]
        T += [A.Normalize(mean=mean, std=std), ToTensorV2()]  # Normalize and convert to Tensor
        logger.info(prefix + ", ".join(f"{x}".replace("always_apply=False, ", "") for x in T if x.p))
        return A.Compose(T)

    except ImportError:  # package not installed, skip
        logger.warning(f"{prefix}⚠️ not found, install with `pip install albumentations` (recommended)")
    except Exception as e:
        logger.info(f"{prefix}{e}")


def classify_transforms(size=224):
    # Transforms to apply if albumentations not installed
    assert isinstance(size, int), f"ERROR: classify_transforms size {size} must be integer, not (list, tuple)"
    # T.Compose([T.ToTensor(), T.Resize(size), T.CenterCrop(size), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)])
    return T.Compose([CenterCrop(size), ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)])


class LetterBox:
    # YOLOv5 LetterBox class for image preprocessing, i.e. T.Compose([LetterBox(size), ToTensor()])
    def __init__(self, size=(640, 640), auto=False, stride=32):
        super().__init__()
        self.h, self.w = (size, size) if isinstance(size, int) else size
        self.auto = auto  # pass max size integer, automatically solve for short side using stride
        self.stride = stride  # used with auto

    def __call__(self, im):  # im = np.array HWC
        imh, imw = im.shape[:2]
        r = min(self.h / imh, self.w / imw)  # ratio of new/old
        h, w = round(imh * r), round(imw * r)  # resized image
        hs, ws = (math.ceil(x / self.stride) * self.stride for x in (h, w)) if self.auto else self.h, self.w
        top, left = round((hs - h) / 2 - 0.1), round((ws - w) / 2 - 0.1)
        im_out = np.full((self.h, self.w, 3), 114, dtype=im.dtype)
        im_out[top : top + h, left : left + w] = cv2.resize(im, (w, h), interpolation=cv2.INTER_LINEAR)  # noqa:E203
        return im_out


class CenterCrop:
    # YOLOv5 CenterCrop class for image preprocessing, i.e. T.Compose([CenterCrop(size), ToTensor()])
    def __init__(self, size=640):
        super().__init__()
        self.h, self.w = (size, size) if isinstance(size, int) else size

    def __call__(self, im):  # im = np.array HWC
        imh, imw = im.shape[:2]
        m = min(imh, imw)  # min dimension
        top, left = (imh - m) // 2, (imw - m) // 2
        return cv2.resize(im[top : top + m, left : left + m], (self.w, self.h), interpolation=cv2.INTER_LINEAR)  # noqa:E203


class ToTensor:
    # YOLOv5 ToTensor class for image preprocessing, i.e. T.Compose([LetterBox(size), ToTensor()])
    def __init__(self, half=False):
        super().__init__()
        self.half = half

    def __call__(self, im):  # im = np.array HWC in BGR order
        im = np.ascontiguousarray(im.transpose((2, 0, 1))[::-1])  # HWC to CHW -> BGR to RGB -> contiguous
        im = torch.from_numpy(im)  # to torch
        im = im.half() if self.half else im.float()  # uint8 to fp16/32
        im /= 255.0  # 0-255 to 0.0-1.0
        return im


def flip_landmarks_and_occlusions_lr(labels):
    def apply_offset(index, offsets):
        return tuple([index + x for x in offsets])

    IDX_LANDMARKS_x = apply_offset(IDX_LANDMARKS, (0, 2, 4, 6, 8))
    labels[:, IDX_LANDMARKS_x] = 1 - labels[:, IDX_LANDMARKS_x]

    IDX_LANDMARKS_lr = apply_offset(IDX_LANDMARKS, (0, 1, 2, 3, 6, 7, 8, 9))
    IDX_LANDMARKS_rl = apply_offset(IDX_LANDMARKS, (2, 3, 0, 1, 8, 9, 6, 7))
    labels[:, IDX_LANDMARKS_lr] = labels[:, IDX_LANDMARKS_rl]

    IDX_EYE_STATUS_lr = apply_offset(IDX_EYE_STATUS, (0, 1))
    IDX_EYE_STATUS_rl = apply_offset(IDX_EYE_STATUS, (1, 0))
    labels[:, IDX_EYE_STATUS_lr] = labels[:, IDX_EYE_STATUS_rl]

    IDX_OCCLUSIONS_lr = apply_offset(IDX_OCCLUSIONS, (0, 1, 4, 5))
    IDX_OCCLUSIONS_rl = apply_offset(IDX_OCCLUSIONS, (1, 0, 5, 4))
    labels[:, IDX_OCCLUSIONS_lr] = labels[:, IDX_OCCLUSIONS_rl]
    return labels


def random_smear_xyxy(image, labels, p, shift=0.08, shift_scale=0.02, alpha=0.4, alpha_scale=0.05):
    if len(labels) > 0:
        h, w, _ = image.shape
        for i, lb in enumerate(labels):
            if random.random() < p:
                # copy and stack
                x1, y1, x2, y2 = [int(x) for x in lb[1:5]]
                shift_x = (shift + np.random.normal() * shift_scale) * (x2 - x1)
                shift_y = (shift + np.random.normal() * shift_scale) * (y2 - y1) * 0.25
                if random.random() < 0.5:
                    shift_x *= -1
                if random.random() < 0.5:
                    shift_y *= -1
                x1s = min(max(x1 + shift_x, 0), w - 1)
                y1s = min(max(y1 + shift_y, 0), h - 1)
                x2s = min(max(x2 + shift_x, 0), w - 1)
                y2s = min(max(y2 + shift_y, 0), h - 1)
                x1s, y1s, x2s, y2s = [int(x) for x in [x1s, y1s, x2s, y2s]]
                real_alpha = alpha + np.random.normal() * alpha_scale
                image[y1s:y2s, x1s:x2s] = image[y1s:y2s, x1s:x2s] * (1 - real_alpha) + image[y1 : y1 + y2s - y1s, x1 : x1 + x2s - x1s] * real_alpha
                # Adjust blur label
                labels[i, IDX_BLUR] = 1
    return image, labels
