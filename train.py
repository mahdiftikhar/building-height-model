import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms as T
from torchsummary import summary
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from pprint import pprint
import copy
import numpy as np
import os
import argparse
from datetime import datetime

# import yaml


from util.util import train_test_split

# ? remove printing of warnings5
# import warnings
# warnings.filterwarnings("ignore")

from model.dataset import CroppedDataset, cast_to_device
from model.layers import Model
from model.loss import RMSELoss, combining_loss


DATASET_PATH = "dataset.csv"
IMAGE_DIR = "images"
IMAGE_SIZE = 640
BATCH_SIZE = 128


def train_cropped(
    model, data_loaders: dict, optimizer, loss_fn, writer, num_epochs=10, device="cpu"
):
    ...
    print("TRAINING STARTED")

    val_loss_history = []
    train_loss_history = []
    best_model_wts = copy.deepcopy(model.state_dict())
    last_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 1000000
    counter = 0

    time_str = datetime.now().strftime("%d_%m_%Y_%H_%M_%S")
    os.mkdir(f"weights/{time_str}")

    for epoch in tqdm(range(num_epochs)):
        print(f"Epoch {epoch} / {num_epochs - 1}", end="\t")

        for phase in ["train", "val"]:
            if phase == "train":
                model.train()
            elif phase == "val":
                model.eval()

            running_shd_loss = 0.0
            running_height_loss = 0.0

            for x in tqdm(data_loaders[phase]):
                counter += 1

                image, labels_shd_len, labels_height, solor_angle = cast_to_device(
                    x, device
                )

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == "train"):
                    pred_shd_len, pred_height = model(image, solor_angle)
                    pred_shd_len = pred_shd_len.squeeze()
                    pred_height = torch.clip(pred_height, 0, 33)
                    pred_height = pred_height.squeeze()

                    shd_loss = loss_fn(pred_shd_len, labels_shd_len)
                    height_loss = loss_fn(pred_height, labels_height)
                    combinined_loss = combining_loss(shd_loss, height_loss)

                    print(
                        pred_shd_len,
                        labels_shd_len,
                        (pred_shd_len - labels_shd_len) ** 2,
                    )

                    if shd_loss == np.nan:
                        print(pred_shd_len, labels_shd_len)

                    if phase == "train":
                        combinined_loss.backward()
                        torch.nn.utils.clip_grad_norm_(
                            model.parameters(), max_norm=10, norm_type=1
                        )
                        optimizer.step()

                    writer.add_scalar(
                        f"Loss Shadow Length/{phase} fast", shd_loss.item(), counter
                    )
                    writer.add_scalar(
                        f"Loss Height/{phase} fast", height_loss.item(), counter
                    )
                    # print(f"Loss Shadow Length/{phase}", shd_loss.item(), epoch)

                    running_shd_loss += shd_loss.item()
                    running_height_loss += height_loss.item()

            shd_epoch_loss = running_shd_loss / len(data_loaders[phase].dataset)
            height_epoch_loss = running_height_loss / len(data_loaders[phase].dataset)

            writer.add_scalar(f"Loss Shadow Length/{phase}", shd_epoch_loss, epoch)
            writer.add_scalar(f"Loss Height/{phase}", height_epoch_loss, epoch)

            print(f"{phase} loss: {shd_epoch_loss:.4f}", end="\t")
            print(f"{phase} height loss: {height_epoch_loss:.4f}", end="\t")

            if phase == "val" and shd_epoch_loss < best_loss:
                best_loss = shd_epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(best_model_wts, os.path.join("weights", time_str, "best.pt"))

            if phase == "val":
                val_loss_history.append(shd_epoch_loss)
                last_model_wts = copy.deepcopy(model.state_dict())
                torch.save(last_model_wts, os.path.join("weights", time_str, "last.pt"))

            if phase == "train":
                train_loss_history.append(shd_epoch_loss)

        print()

    print("-" * 30)
    print(f"Training Complete")
    print(f"Best Validation Loss: {best_loss:.4f}")

    return val_loss_history, train_loss_history


def main(args):
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    df = pd.read_csv(args.data)
    train_df, val_df = train_test_split(df)

    transforms = T.Compose([T.ToTensor(), T.Resize((50, 50))])
    train_dataset = CroppedDataset(train_df, IMAGE_DIR, transforms)
    val_dataset = CroppedDataset(val_df, IMAGE_DIR, transforms)

    BATCH_SIZE = args.batch_size

    dataloaders = {
        "train": DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True),
        "val": DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True),
    }

    model = Model().to(device)

    if args.optimizer == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    elif args.optimizer == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
    else:
        raise ValueError("Optimizer not supported")

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

    if args.multi_gpu:
        model = torch.nn.DataParallel(model, device_ids=[0, 1, 2, 3])

    writer = SummaryWriter()

    _, _ = train_cropped(
        model,
        dataloaders,
        optimizer,
        loss_fn,
        writer,
        num_epochs=args.epochs,
        device=device,
    )

    writer.flush()
    writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="train", description="Training the model")
    parser.add_argument("--gpu", type=int, help="GPU number", default=0, required=False)
    parser.add_argument(
        "--data",
        type=str,
        help="Path to dataset",
        default="dataset.csv",
        required=False,
    )
    parser.add_argument("--optimizer", type=str, help="Optimizer", default="adam")
    parser.add_argument("--batch_size", type=int, help="Batch size", default=64)
    parser.add_argument("--epochs", type=int, help="Number of epochs", default=50)
    parser.add_argument("--multi-gpu", action="store_true", default=False)
    parser.add_argument("--loss", type=str, help="Loss function", default="l1")

    args = parser.parse_args()

    main(args)
