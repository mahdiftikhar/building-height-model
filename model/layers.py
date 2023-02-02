from torch import nn
from torch import hub


class Yolo(nn.Module):
    def __init__(self, model="yolov5s", pretrained=True):
        super(Yolo, self).__init__()
        self.yolo = hub.load(f"ultralytics/yolov5", model, pretrained=True)
        self.yolo.to("cpu")

    def forward(self, x):
        x = self.yolo(x)
        print(x.pandas())

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
    def __init__(self, yolo_model="yolov5s", yolo_pretrained=True):
        super().__init__()
        self.yolo = Yolo(model=yolo_model, pretrained=yolo_pretrained)
        # self.cropping = Cropping()
        # self.lambdaLayer = Lambda()
        # self.shadow_length = ShadowLength()

    def forward(self, x):
        image, class_id, bbox, shd_len, height, lat, long, time = x
        y = self.yolo(image)

        # x = self.cropping(x)
        # x = self.lambda(x)
        # x = self.shadow_length(x)
        return y
