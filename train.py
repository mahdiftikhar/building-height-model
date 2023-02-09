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


# import warnings

# warnings.filterwarnings("ignore")

from model.dataset import CroppedDataset
from model.layers import Model, ShadowLength


DATASET_PATH = "dataset.csv"
IMAGE_DIR = "images"
INPUT_SHAPE = 640
BATCH_SIZE = 32


def train_cropped(
    model, data_loaders: dict, optimizer, loss_fn, writer, num_epochs=10, device="cpu"
):
    ...
    print("TRAINING 

    val_loss_history = []
    train_loss_history = []
    best_model_wts = copy.deepcopy(model.state_dict())
    last_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 1000000

    for epoch in tqdm(range(num_epochs)):
        print(f"Epoch {epoch} / {num_epochs - 1}", end="\t")

        for phase in ["train", "val"]:
            if phase == "train":
                model.train()
            elif phase == "val":
                model.eval()

            running_loss = 0.0

            for x in tqdm(data_loaders[phase]):
                image = x.image
                shd_len = x.shd_len.view(-1, 1)

                image = image.float().to(device)
                shd_len = shd_len.float().to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(image)

                    loss = loss_fn(outputs, shd_len)

                    if loss == np.nan:
                        print(outputs, shd_len)

                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                    writer.add_scalar(f"Loss/{phase}", loss, epoch)

                running_loss += np.nan_to_num(loss.item())

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


def other_train(
    model, train_dataloader, optimizer, criterion, epochs=5, device="gpu", writer=None
):
    for epoch in tqdm(range(epochs)):
        # set the running loss at each epoch to zero
        running_loss = 0.0
        # we will enumerate the train loader with starting index of 0
        # for each iteration (i) and the data (tuple of input and labels)
        for i, data in enumerate(train_dataloader):
            inputs = data.image
            labels = data.shd_len
            inputs = inputs.float().to(device)
            labels = labels.float().to(device)

            # clear the gradient
            optimizer.zero_grad()

            # feed the input and acquire the output from network
            outputs = model(inputs)
            # print(outputs, labels)

            # calculating the predicted and the expected loss

            loss = criterion(outputs, labels)

            # compute the gradient
            loss.backward()
            writer.add_scalar("loss/train", loss, epoch)
            # update the parameters
            optimizer.step()

            if loss.item() == np.nan:
                for out, lab in zip(outputs, labels):
                    print(out, lab)

            # print statistics
            # if i % 10 == 0:
            #     print(
            #         "[%d, %5d] loss: %.3f %.3f"
            #         % (epoch + 1, i + 1, running_loss, loss.item())
            #     )
            #     running_loss += loss.item()


def main():
    df = pd.read_csv(DATASET_PATH)
    df = df[df.shadow_length != -1]
    df.reset_index(drop=True, inplace=True)

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

    train_df.reset_index(drop=True, inplace=True)
    val_df.reset_index(drop=True, inplace=True)

    transforms = T.Compose([T.ToTensor(), T.Resize((50, 50))])
    train_dataset = CroppedDataset(train_df, IMAGE_DIR, transforms)
    val_dataset = CroppedDataset(val_df, IMAGE_DIR, transforms)

    dataloaders = {
        "train": DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True),
        "val": DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True),
    }

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = ShadowLength().to(device)
    loss_fn = torch.nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    writer = SummaryWriter()

    val_loss_hist, train_loss_hist = train_cropped(
        model, dataloaders, optimizer, loss_fn, writer, num_epochs=50, device=device
    )

    # other_train(
    #     model,
    #     dataloaders["train"],
    #     optimizer,
    #     loss_fn,
    #     epochs=5,
    #     device=device,
    #     writer=writer,
    # )

    writer.flush()
    writer.close()


if __name__ == "__main__":
    main()
