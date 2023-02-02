from torch import nn
from torch import hub
from yolov5.models.yolo import DetectionModel as YoloModel
from yolov5.models.common import Detections


class Yolo(nn.Module):
    def __init__(self, cfg_path="model/yolo_cfg/yolov5s.yaml", pretrained=True):
        super(Yolo, self).__init__()
        # self.yolo = hub.load(f"ultralytics/yolov5", model, pretrained=True)
        # self.yolo.to("cpu")
        self.yolo = YoloModel(cfg=cfg_path, ch=3)

    def forward(self, x):
        x = self.yolo(x)
        return x


class Cropping(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        pass


class Lambda(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        pass


class ShadowLength(nn.Module):
    def __init__():
        super().__init__()

    def forward(self, x):
        pass


class Model(nn.Module):
    def __init__(self, yolo_cfg="model/yolo_cfg/yolov5s.yaml", yolo_pretrained=True):
        super().__init__()
        self.yolo = Yolo(cfg_path=yolo_cfg, pretrained=yolo_pretrained)
        self.yolo_detections = Detections
        # self.cropping = Cropping()
        # self.lambdaLayer = Lambda()
        # self.shadow_length = ShadowLength()

    def forward(self, x):
        image, class_id, bbox, shd_len, height, lat, long, time = x
        image = image.view((1, 3, 640, 640))

        y = self.yolo(image)

        z = self.yolo_detections([image], y)
        print(type(z))

        # x = self.cropping(x)
        # x = self.lambda(x)
        # x = self.shadow_length(x)
        return y
