import torch
import numpy as np
import src.Modeling.gan as gan_model
import os.path as osp
import torch.nn.functional as F

import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid

from torch_geometric.nn import GCNConv  # noqa

def relabel_minority_and_majority_classes(data):
    uniq_labels = np.unique(data.y, return_counts=True)
    minority_class = np.unique(data.y, return_counts=True)[1].argmax()
    new_y = np.array([0 if i == minority_class else 1 for i in data.y])
    return new_y

def readjust_ratio(x, y):
    """usecase as followed => readjust_ratio(gcn.data.x, gcn.data.y)"""
    from sklearn.model_selection import train_test_split

    ind = np.arange(y.shape[0])
    X_train_ind, X_test_ind, y_train, y_test = train_test_split(ind,y,
                                                                test_size=0.948,
                                                                random_state=1,
                                                                stratify=y)

    # TODO how does argument = y works
    X_val_ind, X_test_ind, y_val, y_test = train_test_split(X_test_ind, y_test,
                                                            test_size=0.4,
                                                            stratify=y_test,
                                                            random_state=1)  # 0.25 x 0.8 = 0.2
    _, X_val_ind, _, y_val = train_test_split(X_val_ind, y_val, test_size=0.4,
                                              stratify=y_val,
                                              random_state=1)  # 0.25 x 0.8 = 0.2
    X_val_ind = X_val_ind[:500]
    X_test_ind = X_test_ind[:1000]

    train_index = X_train_ind
    val_index = X_val_ind
    test_index = X_test_ind

    def from_intind_to_boolind():
        ind_bool = torch.zeros(y.shape[0]).type(torch.ByteTensor)
        ind_bool[train_index] = 1
        train_mask_bool = ind_bool
        ind_bool = torch.zeros(y.shape[0]).type(torch.ByteTensor)
        ind_bool[test_index] = 1
        test_mask_bool = ind_bool
        ind_bool = torch.zeros(y.shape[0]).type(torch.ByteTensor)
        ind_bool[val_index] = 1
        val_mask_index = ind_bool
        return train_mask_bool, test_mask_bool, val_mask_index

    return from_intind_to_boolind()

if __name__ == '__main__':
    # dtype = torch.float
    # device = torch.device("gpu")

    #=====================
    #==torch_geometric
    #=====================

    # dataset = 'Cora'
    # path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'Data',
    #                 'External')
    # dataset = Planetoid(path, dataset, T.NormalizeFeatures())
    # data = dataset[0]
    # # relabel_minority_and_majority_classes(data)
    # from torch_geometric.data import DataLoader
    # dataloader = DataLoader(dataset)  # not re-labeled

    #=====================
    #==dgl
    #=====================

    data, _ = torch.load(r'C:\Users\Anak\PycharmProjects\AdaptiveGraphStructureEmbedding\Notebook\data\Cora\Cora\processed\data.pt')
    new_y = relabel_minority_and_majority_classes(data)
    data.y = new_y
    data.num_classes = np.unique(data.y).shape[0]


    gan = gan_model.GAN(data)

    new_y = relabel_minority_and_majority_classes(gan.data)
    # select only minority data
    new_y = new_y[np.where(new_y==0)]
    data.x = data.x[np.where(new_y==0)]
    data.y = new_y
    data.num_classes = np.unique(data.y).shape[0]



    # # n_features = 1433
    # gan.data.train_mask, gan.data.test_mask, gan.data.val_mask = readjust_ratio(gan.data.x, gan.data.y)

    # =====================
    # ==for gan
    # =====================
    gan.run_gan()
    # gan.save_loss_to_file()


