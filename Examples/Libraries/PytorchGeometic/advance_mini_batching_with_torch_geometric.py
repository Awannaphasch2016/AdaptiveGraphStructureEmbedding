# Note that 2 different nodes types must be iterated with different number of nodes wrt its totoal nodes type
import torch
from torch_geometric.data import Data, DataLoader

# TODO here>> create DataLoder for Cora using Plantoid data
import os.path as osp
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T

from collections import Counter
from sklearn.datasets import make_moons
from imblearn.datasets import make_imbalance

dataset = 'Cora'
path = osp.join(osp.dirname(osp.realpath(__file__)), '..', '..', 'Data', 'External')
dataset = Planetoid(path, dataset, T.NormalizeFeatures())
data = dataset[0]

from torch_geometric.data import DataLoader
import numpy as np
dataloader = DataLoader(dataset)

# Batch(batch=[2708], edge_index=[2, 10556], test_mask=[2708], train_mask=[2708], val_mask=[2708], x=[2708, 1433], y=[2708])
for batch_, edge_index_, test_mask_, train_mask_, val_mask_, x_, y_ in dataloader:
    batch, edge_index, x ,y = batch_[1], edge_index_[1], x_[1], y_[1]
    test_mask, train_mask, val_mask = test_mask_[1], train_mask_[1], val_mask_[1]

def ratio_func(y, multiplier, minority_class):
    target_stats = Counter(y)
    return {minority_class: int(multiplier * target_stats[minority_class])}

min_ind = np.unique(y, return_counts=True)[1].argmin()
y = y.numpy()

#=====================
#==make cora imbalance
#=====================
ind_x = np.arange(x.shape[0] ).reshape(-1,1)

ind_x_, y_ = make_imbalance(ind_x, y, sampling_strategy=ratio_func,
                        **{"multiplier": 0.2,
                           "minority_class": 6}, )
ind_x = ind_x.flatten()
print()

#=====================
#==labeling only 5 percent of the dataset
#=====================
# shuffled_ind = np.random.shuffle(ind_x_)
# percentage = 0.05
# num_select = int(y.shape[0] * percentage)
# selected_ind = shuffled_ind[:num_select]
# selected_ind_x = x[selected_ind]
# selected_ind_y = y[selected_ind]

#=====================
#== train_test_split
#=====================
# import numpy as np
# from sklearn.model_selection import train_test_split
# X, y = np.arange(10).reshape((5, 2)), range(5)
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.33, random_state=42)



