import torch
import numpy as np



def relabel_minority_and_majority_classes(data, isLog=False):

    uniq_labels = np.unique(data.y, return_counts=True)
    minority_class = np.unique(data.y, return_counts=True)[1].argmax()
    new_y = np.array([0 if i == minority_class else 1 for i in data.y])
    return new_y

def preparing_cora_for_new_purposed_model(isLog=False):

    data, _ = torch.load(
        r'C:\Users\Anak\PycharmProjects\AdaptiveGraphStructureEmbedding\Notebook\data\Cora\Cora\processed\data.pt')
    new_y = relabel_minority_and_majority_classes(data, isLog=isLog)
    data.y = new_y
    data.num_classes = np.unique(data.y).shape[0]
    return data
