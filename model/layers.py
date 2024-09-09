from torch import nn
from torch.functional import F
from torch import hub
import torch
from torchsummary import summary
import numpy as np


# from yolov5.models.yolo import DetectionModel as YoloModel
# from yolov5.models.common import Detections
from model.utils import its_xyxy_time, its_denormalize_time

from torchvision.transforms import Pad, Resize
from torchvision.models import resnet101, resnet18, resnet50


DEFAULT_IMAGE_SIZE = 640

# class Yolo(nn.Module):
#     def __init__(self, cfg_path="model/yolo_cfg/yolov5s.yaml", pretrained=True):
#         super(Yolo, self).__init__()
#         self.yolo = YoloModel(cfg=cfg_path, ch=3)

#     def forward(self, x):
#         x = self.yolo(x)
#         return x


class Cropping(nn.Module):
    def __init__(self, out_shape=100):
        super().__init__()
        self.out_shape = out_shape

    def forward(self, x, bbox):
        # ? the following function should be written to work with cxcywh format as well as the xywh format
        # ? becuase i don't know how i'm going to receive the bounding box from shayaan

        # ? for now only catering to the format available from ground truth annotations
        image_shape = x.shape[2:]

        xyxy = its_xyxy_time(its_denormalize_time(bbox, image_shape))

        cropped_images = []

        for bbox in xyxy:
            cropped = x[:, :, bbox[1] : bbox[3], bbox[0] : bbox[2]]
            # padded = Pad(20)(cropped)

            resized = Resize((self.out_shape, self.out_shape))(cropped)
            cropped_images.append(resized)

        # test = cropped_images[8]
        # test = test.permute(3, 2, 1, 0).view(50, 50, 3).numpy()

        # import matplotlib.pyplot as plt

        return cropped_images


class Lambda(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, shd_len, solar_angle):
        # shd_len = shd_len * 1000
        # shd_len = shd_len.view(solar_angle.shape)
        height = shd_len * torch.tan(solar_angle)
        height = torch.nan_to_num(height, nan=100)
        # height[torch.isnan(height)] = torch.finfo(height.dtype).max

        return height

class CustomActivation(nn.Module):
    def __init__(self):
        super().__init__()
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        
        
        x_shd = x[:, 0]
        x_solar = x[:, 1]

        # x_shd = self.relu(x_shd)
        x_shd = self.sigmoid(x_shd) * 1000

        # print("\n=========================")
        # print(x_solar)
        # print("=========================")

        # custom relu for solar angle
        # x_solar = torch.clip(x_solar, 0.1, torch.pi / 2)
        x_solar = self.sigmoid(x_solar) * torch.pi / 2


        x_shd = x_shd.view(x_shd.shape[0], 1)
        x_solar = x_solar.view(x_solar.shape[0], 1)
        x = torch.cat((x_shd, x_solar), dim=1)
        return x

class ShadowLength(nn.Module):
    def __init__(self, backbone="resnet18", pretrained=False):
        super().__init__()

        if backbone == "resnet18":
            self.resnet = nn.Sequential(
                *list(resnet18(pretrained=pretrained).children())[:-1],
                nn.Flatten(),
                # nn.Linear(in_features=512, out_features=256, bias=True),
                # nn.ReLU(),
                # nn.Linear(in_features=256, out_features=64, bias=True),
                # nn.ReLU(),
                nn.Linear(in_features=512, out_features=2, bias=True),
                CustomActivation()
            )
        elif backbone == "resnet101":
            self.resnet = nn.Sequential(
                *list(resnet101(pretrained=pretrained).children())[:-1],
                nn.Flatten(),
                nn.Linear(in_features=2048, out_features=512, bias=True),
                nn.Linear(in_features=512, out_features=256, bias=True),
                nn.ReLU(),
                nn.Linear(in_features=256, out_features=64, bias=True),
                nn.ReLU(),
                nn.Linear(in_features=64, out_features=2, bias=True),
                CustomActivation()
            )
        elif backbone == "resnet50":
            self.resnet = nn.Sequential(
                *list(resnet50(pretrained=pretrained).children())[:-1],
                nn.Flatten(),
                # nn.Linear(in_features=2048, out_features=512, bias=True),
                # nn.Linear(in_features=512, out_features=256, bias=True),
                # nn.ReLU(),
                # nn.Linear(in_features=256, out_features=64, bias=True),
                # nn.ReLU(),
                nn.Linear(in_features=2048, out_features=2, bias=True),
               CustomActivation()
            )

    def forward(self, x):
        x = self.resnet(x)
        x_shd = x[:, 0]
        x_solar = x[:, 1]
        return x_shd, x_solar


class Model(nn.Module):
    def __init__(
        self,
        yolo_cfg="model/yolo_cfg/yolov5s.yaml",
        yolo_pretrained=True,
        shd_len_backbone="resnet50",
        pretrained=False,
    ):
        super().__init__()
        # self.yolo = Yolo(cfg_path=yolo_cfg, pretrained=yolo_pretrained)
        self.lambdaLayer = Lambda()
        self.shadow_length = ShadowLength(
            backbone=shd_len_backbone, pretrained=pretrained
        )

    def forward(self, image):
        shd_len, solar_angle = self.shadow_length(image)
        # print()
        # print("shadow length", shd_len)
        # print("solar angle", solar_angle)
        height = self.lambdaLayer(shd_len, solar_angle)
        # print("height", height)

        # print("")

        return shd_len, solar_angle, height
