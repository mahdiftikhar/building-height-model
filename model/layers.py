import torch
from torch import nn
from torch import hub
# from yolov5.models.yolo import DetectionModel as YoloModel
# from yolov5.models.common import Detections

from pathlib import Path
from yolov7.models.yolo import Model
from yolov7.utils.google_utils import attempt_download
from yolov7.utils.torch_utils import select_device

def createYolo(name, pretrained, channels, classes, autoshape):
    """Creates a specified model

    Arguments:
        name (str): name of model, i.e. 'yolov7'
        pretrained (bool): load pretrained weights into the model
        channels (int): number of input channels
        classes (int): number of model classes

    Returns:
        pytorch model
    """
    try:
        cfg = list((Path(__file__).parent / 'cfg').rglob(f'{name}.yaml'))[0]  # model.yaml path
        model = Model(cfg, channels, classes)
        if pretrained:
            fname = f'{name}.pt'  # checkpoint filename
            attempt_download(fname)  # download if not found locally
            ckpt = torch.load(fname, map_location=torch.device('cpu'))  # load
            msd = model.state_dict()  # model state_dict
            csd = ckpt['model'].float().state_dict()  # checkpoint state_dict as FP32
            csd = {k: v for k, v in csd.items() if msd[k].shape == v.shape}  # filter
            model.load_state_dict(csd, strict=False)  # load
            if len(ckpt['model'].names) == classes:
                model.names = ckpt['model'].names  # set class names attribute
        if autoshape:
            model = model.autoshape()  # for file/URI/PIL/cv2/np inputs and NMS
        device = select_device('0' if torch.cuda.is_available() else 'cpu')  # default to GPU if available
        return model.to(device)

    except Exception as e:
        s = 'Cache maybe be out of date, try force_reload=True.'
        raise Exception(s) from e

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

if __name__ == '__main__':
    # model = custom(path_or_model='yolov7.pt')  # custom example
    model = createYolo(name='yolov7-tiny', pretrained=True, channels=3, classes=80, autoshape=True)  # pretrained example
    
    print(Path(__file__).parent)