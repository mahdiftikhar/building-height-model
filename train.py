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


from util.util import train_test_split


# import warnings

# warnings.filterwarnings("ignore")

from model.dataset import CroppedDataset, cast_to_device
from model.layers import Model


DATASET_PATH = "dataset.csv"
IMAGE_DIR = "images"
IMAGE_SIZE = 640
BATCH_SIZE = 64


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

    for epoch in tqdm(range(num_epochs)):
        print(f"Epoch {epoch} / {num_epochs - 1}", end="\t")

        for phase in tqdm(["train", "val"]):
            if phase == "train":
                model.train()
            elif phase == "val":
                model.eval()

            running_loss = 0.0

            for x in tqdm(data_loaders[phase]):
                image, labels_shd_len, labels_height, solor_angle = cast_to_device(
                    x, device
                )

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == "train"):
                    pred_shd_len, height = model(image, solor_angle)
                    pred_shd_len = pred_shd_len.squeeze()

                    shd_loss = loss_fn(pred_shd_len, labels_shd_len)
                    height_loss = (height - labels_height).abs().mean()

                    if shd_loss == np.nan:
                        print(pred_shd_len, labels_shd_len)

                    if phase == "train":
                        shd_loss.backward()
                        optimizer.step()

                    writer.add_scalar(f"Loss Shadow Length/{phase}", shd_loss, epoch)
                    writer.add_scaler(f"Loss Height/{phase}", height_loss, epoch)

                running_loss += np.nan_to_num(shd_loss.item())

                epoch_loss = running_loss / len(data_loaders[phase].dataset)

            # print(f"{phase} loss: {epoch_loss:.4f}", end="\t")

            if phase == "val" and epoch_loss < best_loss:
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(best_model_wts, os.path.join("weights", "best.pt"))

            if phase == "val":
                val_loss_history.append(epoch_loss)
                last_model_wts = copy.deepcopy(model.state_dict())
                torch.save(last_model_wts, os.path.join("weights", "last.pt"))

            if phase == "train":
                train_loss_history.append(epoch_loss)

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

    dataloaders = {
        "train": DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True),
        "val": DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True),
    }

    model = Model().to(device)
    loss_fn = torch.nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    writer = SummaryWriter()

    _, _ = train_cropped(
        model, dataloaders, optimizer, loss_fn, writer, num_epochs=50, device=device
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

    args = parser.parse_args()

    main(args)
