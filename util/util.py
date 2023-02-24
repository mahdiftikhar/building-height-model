import os
import json
from pprint import pprint

def train_test_split(df, split=0.8):
    size = len(df)
    train_df = df[: int(size * split)]
    val_df = df[int(size * split) :]

    train_df.reset_index(drop=True, inplace=True)
    val_df.reset_index(drop=True, inplace=True)

    return train_df, val_df

def write_eval_file(shd_loss, height_loss, analytical_height_loss, shd_batch_losses, height_batch_losses, analytical_height_batch_losses, args):
    obj = {
            "Model": args.model,
            "Loss fn": args.loss,
            "Weights File": args.weights,
            "CSV file": args.data,
            "shd_loss": shd_loss,
            "height_loss": height_loss,
            "analytical_height_loss": analytical_height_loss,
            "shd_batch_losses": shd_batch_losses,
            "height_batch_losses": height_batch_losses,
            "analytical_height_batch_losses": analytical_height_batch_losses,
        }
    
    pprint(obj, width=1)
        
    folder = "/".join(args.weights.split("/")[0:2])
    folderFiles = os.listdir(folder)
    evalFiles = len([file for file in folderFiles if "eval" in file])

    fileName = f"eval-{evalFiles+1}.json"
    with open(f"{folder}/{fileName}", "w") as f:
        json.dump(obj, f, indent=4)

def write_train_file(model, optimizer, loss_fn, num_epochs, shd_loss_weight, folder):
    obj = {
            "Model": model,
            "Optimizer": optimizer,
            "Loss fn": loss_fn,
            "Num Epochs": num_epochs,
            "shd_loss_weight": shd_loss_weight,
        }
    
    fileName = "train_parameters.json"
    with open(f"{folder}/{fileName}", "w") as f:
        json.dump(obj, f, indent=4)