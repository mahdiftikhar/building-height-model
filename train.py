import torch
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from torchsummary import summary
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from pprint import pprint

from model.dataset import BuildingDataset
from model.layers import Model


DATASET_PATH = "dataset.csv"
IMAGE_DIR = "images"
INPUT_SHAPE = (640, 640)


def train(model, data_loaders: dict, optimizer, loss_fn, num_epochs=10, device="cpu"):
    ...
    print("TRAINING STARTED")

    model.eval()

    for epoch in tqdm(range(num_epochs)):
        for x in data_loaders["train"]:
            # image = image.to(device)
            results = model(x)
            print(results[0].shape)
            print(results[1][0].shape)
            return


if __name__ == "__main__":
    df = pd.read_csv(DATASET_PATH)
    print("Total Images in Dataset:                     ", len(df.image_id.unique()))
    print("Total Bounding Boxes in Dataset:             ", len(df))
    print(
        "Total bounding boxes with shadow annotation: ", len(df[df.shadow_length != -1])
    )

    size = len(df)
    train_df = df[: int(size * 0.8)]
    val_df = df[int(size * 0.8) :]

    print(
        "Total Images in Train Dataset:               ", len(train_df.image_id.unique())
    )
    print(
        "Total Images in Train Dataset:               ", len(val_df.image_id.unique())
    )

    t = transforms.Compose([transforms.ToTensor(), transforms.Resize(INPUT_SHAPE)])

    train_dataset = BuildingDataset(train_df, IMAGE_DIR, t)
    test_dataset = BuildingDataset(val_df, IMAGE_DIR, t)

    # dataloaders = {
    #     "train": DataLoader(train_dataset, batch_size=2, shuffle=False),
    #     "val": DataLoader(test_dataset, batch_size=2, shuffle=False),
    # }

    dataloaders = {"train": train_dataset, "val": test_dataset}

    model = Model()

    loss_fn = None
    optimizer = None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train(model, dataloaders, loss_fn, optimizer, num_epochs=1)
