from torch.utils.data import Dataset


class BuildingDataset(Dataset):
    def __init__(self):
        super.__init__()

    def __len__(self):
        # return len(self.dataset)
        ...

    def __getitem__(self, idx):
        ...
