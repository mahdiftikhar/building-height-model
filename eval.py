import argparse
import os
from tqdm import tqdm
import torch
from torchvision import transforms as T
import pandas as pd
from torch.utils.data import DataLoader
from datetime import datetime
import json

import cv2
from pvlib.solarposition import get_solarposition
import numpy as np

from model.layers import Model
from model.dataset import CroppedDataset, cast_to_device
from model.loss import RMSELoss
from util.util import write_eval_file

DATASET_PATH = "dataset.csv"
IMAGE_DIR = "images"
IMAGE_SIZE = 640
BATCH_SIZE = -4

def eval(model, dataloader, device, loss_fn):
    print("EVALUATION STARTED")
    model.eval()
    shd_running_loss = 0.0
    height_running_loss = 0.0
    analytical_height_running_loss = 0.0
    
    shd_batch_losses = []
    height_batch_losses = []
    analytical_height_batch_losses = []

    for x in tqdm(dataloader, total=len(dataloader)):
        image, labels_shd_len, labels_height, solar_angle = cast_to_device(x, device)
        analytical_height = torch.clip(labels_shd_len / torch.tan(solar_angle), 0, 33)

        pred_shd_len, pred_height = model(image, solar_angle)
        
        pred_shd_len = pred_shd_len.squeeze()
        pred_height = pred_height.squeeze()

        shd_loss = loss_fn(pred_shd_len, labels_shd_len)
        height_loss = loss_fn(pred_height, labels_height)
        analytical_height_loss = loss_fn(pred_height, analytical_height)

        shd_running_loss += shd_loss.item()
        height_running_loss += height_loss.item()
        analytical_height_running_loss += analytical_height_loss.item()

        shd_batch_losses.append(shd_loss.item())
        height_batch_losses.append(height_loss.item())
        analytical_height_batch_losses.append(analytical_height_loss.item())

    shd_running_loss /= len(dataloader)
    height_running_loss /= len(dataloader)
    analytical_height_running_loss /= len(dataloader)

    print(f"SHD loss: {shd_running_loss}")
    print(f"Height loss: {height_running_loss}")
    print(f"Analytical Height loss: {analytical_height_running_loss}")

    return shd_running_loss, height_running_loss, analytical_height_running_loss, shd_batch_losses, height_batch_losses, analytical_height_batch_losses

def main(args):
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    BATCH_SIZE = args.batch_size

    test_df = pd.read_csv(args.data)
    transforms = T.Compose([T.ToTensor(), T.Resize((50, 50))])
    test_dataset = CroppedDataset(test_df, IMAGE_DIR, transforms)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = Model(shd_len_backbone=args.model).to(device)
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

    shd_loss, height_loss, analytical_height_loss, shd_batch_losses, height_batch_losses, analytical_height_batch_losses = eval(model, test_dataloader, device, loss_fn)
    write_eval_file(shd_loss, height_loss, analytical_height_loss, shd_batch_losses, height_batch_losses, analytical_height_batch_losses, args)
    

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
    parser.add_argument("--model", type=str, help="Model name", default="resnet101", required=False)

    args = parser.parse_args()
    print("--------------------")
    print(args)
    print("--------------------")
    main(args)

'''
    Example usage:
        python eval.py --gpu 0 --data PLEASE_WORK.csv --loss l1 --weights weights/22_02_2023_09_07_47-adam-mse/best.pt --model resnet18
'''
