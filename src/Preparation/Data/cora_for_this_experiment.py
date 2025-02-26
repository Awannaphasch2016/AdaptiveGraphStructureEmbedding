import torch
import numpy as np
from torch.utils.data import Dataset


class CoraTorchDataset(Dataset):

    def __init__(self, data, y):
        self.data = data
        self.y = y

    def __len__(self):
        return self.data.size(0)

    def __getitem__(self, idx):
        return self.data[idx], self.y[idx]

def relabel_minority_and_majority_classes(data, isLog=False):

    uniq_labels = np.unique(data.y, return_counts=True)
    minority_class = np.unique(data.y, return_counts=True)[1].argmin()
    new_y = np.array([0 if i == minority_class else 1 for i in data.y])
    return new_y

def preparing_cora_for_new_purposed_model(isLog=False):

    data, _ = torch.load(
        r'C:\Users\Anak\PycharmProjects\AdaptiveGraphStructureEmbedding\Notebook\data\Cora\Cora\processed\data.pt')
    data.y_before_relabel = data.y
    new_y = relabel_minority_and_majority_classes(data, isLog=isLog)
    data.y = new_y
    data.num_classes = np.unique(data.y).shape[0]
    return data
