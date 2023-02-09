from torch import nn
from torch.functional import F
from torch import hub

# from yolov5.models.yolo import DetectionModel as YoloModel
# from yolov5.models.common import Detections
from model.utils import its_xyxy_time, its_denormalize_time

from torchvision.transforms import Pad, Resize
from torchvision.models import resnet101, resnet18, resnet50


# class Yolo(nn.Module):
#     def __init__(self, cfg_path="model/yolo_cfg/yolov5s.yaml", pretrained=True):
#         super(Yolo, self).__init__()
#         self.yolo = YoloModel(cfg=cfg_path, ch=3)

#     def forward(self, x):
#         x = self.yolo(x)
#         return x


class Cropping(nn.Module):
    def __init__(self, out_shape=50):
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

    def forward(self, x):
        pass


class ShadowLength(nn.Module):
    def __init__(self):
        super().__init__()
        # self.resent = resnet101(pretrained=True)
        self.resnet = nn.Sequential(
            *list(resnet50(pretrained=True).children())[:-1],
            nn.Flatten(),
            nn.Linear(in_features=2048, out_features=512, bias=True),
            # nn.Linear(in_features=1024, out_features=512, bias=True),
            # nn.Linear(in_features=512, out_features=64, bias=True),
            # nn.Linear(in_features=64, out_features=16, bias=True),
            nn.Linear(in_features=512, out_features=256, bias=True),
            nn.Sigmoid(),
            nn.Linear(in_features=256, out_features=64, bias=True),
            nn.Sigmoid(),
            nn.Linear(in_features=64, out_features=1, bias=True)
        )

        print(self.resnet)

    def forward(self, x):
        return F.relu(self.resnet(x))


class Model(nn.Module):
    def __init__(self, yolo_cfg="model/yolo_cfg/yolov5s.yaml", yolo_pretrained=True):
        super().__init__()
        # self.yolo = Yolo(cfg_path=yolo_cfg, pretrained=yolo_pretrained)
        # self.lambdaLayer = Lambda()
        # self.shadow_length = ShadowLength()

    def forward(self, x):
        image, class_id, bbox, shd_len, height, lat, long, time = x
        image = image.view((1, 3, 640, 640))

        # y = self.yolo(image)

        x = self.cropping(image, bbox)
        # print(x.shape)

        # x = self.cropping(x)
        # x = self.lambda(x)
        # x = self.shadow_length(x)
        return x
