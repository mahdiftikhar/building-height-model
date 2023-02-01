from torch import nn


class Yolo(nn.module):
    def __init__(self):
        super.__init__()

    def forward(x):
        pass


class Cropping(nn.module):
    def __init__(self):
        super.__init__()

    def forward(x):
        pass


class Lambda(nn.module):
    def __init__(self):
        super.__init__()

    def forward(x):
        pass


class ShadowLength(nn.module):
    def __init__(self):
        super.__init__()

    def forward(x):
        pass


class Model(nn.module):
    def __init__(self):
        super.__init__()
        self.yolo = Yolo()
        self.cropping = Cropping()
        self.lambdaLayer = Lambda()
        self.shadow_length = ShadowLength()

    def forward(x):
        # x = self.yolo(x)
        # x = self.cropping(x)
        # x = self.lambda(x)
        # x = self.shadow_length(x)
        return x
