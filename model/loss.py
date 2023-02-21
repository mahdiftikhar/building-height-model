import torch


def combining_loss(shd_loss, height_loss, shd_loss_weight=1):
    return shd_loss_weight * shd_loss + (1 - shd_loss_weight) * height_loss


class RMSELoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = torch.nn.MSELoss()

    def forward(self, y_pred, y_true):
        return torch.sqrt(self.mse(y_pred, y_true))
