import time
import torch
import torchvision
import torch.nn as nn

from box import xywh2xyxy


class AnchornizedNMS(nn.Module):
    def __init__(self, num_classes, conf_threshold, iou_threshold, max_det=300, max_nms=30000, offset_by_class=True):
        super().__init__()
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.max_det = max_det
        self.max_nms = max_nms
        if num_classes <= 1:
            offset_by_class = False
        self.offset_by_class = offset_by_class
        self.num_classes = num_classes
        self.max_wh = 7680  # (pixels) maximum box width and height

    def forward(self, x):
        # disable nms while training
        if self.training:
            return x
        # x: List[Tensor(N, -1, Channels)]
        # set time limit
        time_limit = 0.3 + 0.03 * x[0].shape[0]  # seconds to quit after
        t = time.time()
        outputs = []
        for batch in x:
            batch = batch[batch[:, 4] > self.conf_threshold]
            if batch.shape[0] == 0:
                continue
            batch = batch[batch[:, 4].argsort(descending=True)[: self.max_nms]]
            boxes, scores = batch[:, :4], batch[:, 4]
            boxes = xywh2xyxy(boxes)
            # scores = objectness * conf
            class_channels = self.num_classes
            if class_channels > 1:
                classes = batch[:, 5 : 5 + class_channels]
                conf, classes = classes.max(1)
                classes = classes.unsqueeze(-1)
                scores *= conf
            else:
                class_channels = 0  # hidden class as 0
                classes = torch.zeros_like(scores).unsqueeze(-1)
            attributes = batch[:, 5 + class_channels :]
            batch = torch.cat((boxes, scores.unsqueeze(-1), classes, attributes), dim=-1)
            # offset by class
            if self.offset_by_class:
                boxes += classes * self.max_wh
            i = torchvision.ops.nms(boxes, scores, self.iou_threshold)  # NMS
            if i.shape[0] > self.max_det:  # limit detections
                i = i[: self.max_det]
            outputs.append(batch[i])
            if (time.time() - t) > time_limit:
                print(f"WARNING: NMS time limit {time_limit:.3f}s exceeded")
                break  # time limit exceeded
        return outputs
