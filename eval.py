import argparse
import os
from tqdm import tqdm
import torch
from torchvision import transforms as T
import pandas as pd
from torch.utils.data import DataLoader

from model.layers import Model
from model.dataset import CroppedDataset, cast_to_device
from model.loss import RMSELoss

DATASET_PATH = "dataset.csv"
IMAGE_DIR = "images"
IMAGE_SIZE = 640
BATCH_SIZE = -4

def eval(model, dataloader, device, loss_fn):
    print("EVALUATION STARTED")
    model.eval()
    shd_running_loss = 0.0
    height_running_loss = 0.0
    for _, x in tqdm(enumerate(dataloader)):
        image, labels_shd_len, labels_height, solor_angle = cast_to_device(x, device)
        pred_shd_len, pred_height = model(image, solor_angle)
        
        pred_shd_len = pred_shd_len.squeeze()
        pred_height = torch.clip(pred_height, 0, 33)
        pred_height = pred_height.squeeze()

        shd_loss = loss_fn(pred_shd_len, labels_shd_len)
        height_loss = loss_fn(pred_height, labels_height)

        shd_running_loss += shd_loss.item()
        height_running_loss += height_loss.item()

    shd_running_loss /= len(dataloader)
    height_running_loss /= len(dataloader)

    print(f"SHD loss: {shd_running_loss}")
    print(f"Height loss: {height_running_loss}")

def main(args):
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    BATCH_SIZE = args.batch_size

    test_df = pd.read_csv(args.data)
    transforms = T.Compose([T.ToTensor(), T.Resize((50, 50))])
    test_dataset = CroppedDataset(test_df, IMAGE_DIR, transforms)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = Model().to(device)
    model.load_state_dict(torch.load(args.weights))

    if args.loss == "l1":
        loss_fn = torch.nn.L1Loss()
    elif args.loss == "mse":
        loss_fn = torch.nn.MSELoss()
    elif args.loss == "smoothl1":
        loss_fn = torch.nn.SmoothL1Loss()
    elif args.loss == "huber":
        loss_fn = torch.nn.HuberLoss()
    elif args.loss == "rmse":
        loss_fn = RMSELoss()
    else:
        raise ValueError("Loss not supported")

    eval(model, test_dataloader, device, loss_fn)
    

if __name__ and "main":
    parser = argparse.ArgumentParser(prog="eval", description="Evaluating the model")
    parser.add_argument("--gpu", type=int, help="GPU number", default=0, required=False)
    parser.add_argument(
        "--data",
        type=str,
        help="Path to dataset",
        default="dataset.csv",
        required=False,
    )
    parser.add_argument("--loss", type=str, help="Loss function", default="rmse", required=False)
    parser.add_argument("--batch_size", type=int, help="Batch size", default=64, required=False)
    parser.add_argument("--weights", type=str, help="Path to weights", default="weights/best.pt", required=False)

    args = parser.parse_args()
    main(args)
