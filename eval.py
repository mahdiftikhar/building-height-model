import argparse
import os
from tqdm import tqdm
import torch
from torchvision import transforms as T
import pandas as pd
from torch.utils.data import DataLoader
from datetime import datetime

import cv2
from pvlib.solarposition import get_solarposition
import numpy as np

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
    for x in tqdm(dataloader, total=len(dataloader)):
        image, labels_shd_len, labels_height, solor_angle = cast_to_device(x, device)
        pred_shd_len, pred_height = model(image, solor_angle)
        
        pred_shd_len = pred_shd_len.squeeze()
        pred_height = pred_height.squeeze()

        shd_loss = loss_fn(pred_shd_len, labels_shd_len)
        height_loss = loss_fn(pred_height, labels_height)

        shd_running_loss += shd_loss.item()
        height_running_loss += height_loss.item()

    shd_running_loss /= len(dataloader)
    height_running_loss /= len(dataloader)

    print(f"SHD loss: {shd_running_loss}")
    print(f"Height loss: {height_running_loss}")

    return shd_running_loss, height_running_loss

def eval_by_csv(model, csv_path, device, loss_fn):
    test_df = pd.read_csv(csv_path)
    prediction_wali_df = pd.read_csv("dataset-handler/analysis.csv")
    transforms = T.Compose([T.ToTensor(), T.Resize((50, 50))])
    model.eval()
    running_loss = 0.0
    for row in tqdm(test_df.iterrows(), total=len(test_df)):
        row = row[1]
        image = transforms(cv2.imread(os.path.join(IMAGE_DIR, row["image"])))
        image = image.unsqueeze(0).to(device)
        
        lat, long = row["lat"], row["long"]
        yr, mo, day, hr = row["time"].split("-")
        yr, mo, day, hr = int(yr), int(mo), int(day), int(hr)
        dt = datetime(yr, mo, day, hour=hr)
        solar_angle = torch.tensor(get_solarposition(dt, lat, long).elevation.values[0])
        solar_angle = solar_angle.to(device)

        pred_shd_len, pred_height = model(image, solar_angle)
        pred_shd_len = pred_shd_len.squeeze()
        pred_height = pred_height.squeeze()

        actual_height = prediction_wali_df.query(f"image == '{row['image']}' and bbox == '{row['bbox']}'")['pred'].values[0]
        
        height_loss = loss_fn(pred_height, torch.tensor(actual_height).to(device))
        running_loss += height_loss.item()

    running_loss /= len(test_df)
    print(f"Height loss: {running_loss}")
    return running_loss

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

    if args.csv:
        print("EVALUATING BY CSV")
        eval_by_csv(model, args.data, device, loss_fn)
    else:
        shd_loss, height_loss = eval(model, test_dataloader, device, loss_fn)

        to_write = f" Model: {args.model}\n Loss fn: {args.loss}\n Weights File: {args.weights}\n CSV file: {args.data}\n Shadow Loss: {shd_loss}\n Height Loss: {height_loss}\n"
        
        fileName = args.weights.split("/")[1]
        csvFile = args.data.split("/")[1].split(".")[0]
        with open(f"weights/{fileName}/eval-{csvFile}.txt", "w") as f:
            f.write(to_write)
    

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
    parser.add_argument("--csv", action="store_true", default=False)

    args = parser.parse_args()
    main(args)

'''
    Example usage:
        python eval.py --gpu 0 --data dataset-handler/small_building_dataset.csv --loss mse --weights weights/22_02_2023_09_07_49/best.pt --model resnet18
'''
