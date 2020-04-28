# import os,sys,inspect
# current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
# parent_dir = os.path.dirname(current_dir)
# sys.path.insert(0,parent_dir)

from src.Modeling.gcn import *
import numpy as np
import pandas as pd
import torch
import torch_geometric

def relabel_minority_and_majority_classes(data):
    uniq_labels = np.unique(data.y, return_counts=True)
    minority_class = np.unique(data.y, return_counts=True)[1].argmax()
    new_y = np.array([0 if i == minority_class else 1 for i in data.y])
    return new_y


def print_ratio():
    gcn.data.y = torch.tensor(gcn.data.y)
    # for name, mask in data('train_mask', 'val_mask', 'test_mask'):
    for name, mask in data('train_mask', 'val_mask', 'test_mask'):
        print(f'---{name}')
        labels, count = np.unique(gcn.data.y[mask].numpy(), return_counts=True)
        ratio = count[1] / count[0]
        print(count)
        # print(ratio)


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
        gcn.data.train_mask = ind_bool
        ind_bool = torch.zeros(y.shape[0]).type(torch.ByteTensor)
        ind_bool[test_index] = 1
        gcn.data.test_mask = ind_bool
        ind_bool = torch.zeros(gcn.data.y.shape[0]).type(torch.ByteTensor)
        ind_bool[val_index] = 1
        gcn.data.val_index = ind_bool

    from_intind_to_boolind()
    print_ratio()



# def init_gcn():
#     # =====================
#     # ==hyper parameters setup
#     # =====================
#     num_batches = 1
#     num_epochs = 200
#
#     # =====================
#     # ==relabeling
#     # =====================
#
#     new_y = relabel_minority_and_majority_classes(data)
#     data.y = new_y
#     data.num_classes = np.unique(data.y).shape[0]
#
#     # # TODO torch dataset
#     # min_class_ind = np.where(data.y == 0)
#     # min_y_data = data.y[min_class_ind]
#     # min_x_data = data.x[min_class_ind]
#
#     # =====================
#     # ==for Gcn
#     # ====================
#     import src.Modeling.gcn as gcn_model
#
#     gcn = gcn_model.GCN(data, dataset)


if __name__ == '__main__':
    #=====================
    #==torch_geometric
    #=====================

    # dataset = 'Cora'
    # path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'Data',
    #                 'External')
    #
    # dataset = Planetoid(path, dataset, T.NormalizeFeatures())
    # data = dataset[0]
    # relabel_minority_and_majority_classes(data)
    #
    # from torch_geometric.data import DataLoader
    #
    # dataloader = DataLoader(dataset)  # not re-labeled
    # new_y = relabel_minority_and_majority_classes(data)
    # data.y = new_y
    # data.num_classes = np.unique(data.y).shape[0]

    #=====================
    #==dgl
    #======================
    # from dgl.data import CoraDataset
    #
    # data = CoraDataset()
    # data.y = data.labels
    # data.x = data.features
    # data.num_features = data.x.shape[1]
    # data.edge_index = np.array(data.graph.to_undirected().edges).T

    data, _ = torch.load(r'C:\Users\Anak\PycharmProjects\AdaptiveGraphStructureEmbedding\Notebook\data\Cora\Cora\processed\data.pt')
    new_y = relabel_minority_and_majority_classes(data)
    data.y = new_y
    data.num_classes = np.unique(data.y).shape[0]

    # relabel_minority_and_majority_classes(data)

    from torch_geometric.data import DataLoader

    new_y = relabel_minority_and_majority_classes(data)
    data.y = new_y
    data.num_classes = np.unique(data.y).shape[0]

    # # TODO torch dataset
    # min_class_ind = np.where(data.y == 0)
    # min_y_data = data.y[min_class_ind]
    # min_x_data = data.x[min_class_ind]

    # =====================
    # ==for Gcn
    # ====================
    import src.Modeling.gcn as gcn_model

    gcn = gcn_model.GCN(data)
    # readjust_ratio(gcn.data.x, gcn.data.y)
    # print_ratio()
    # exit()
    gcn.data.y = torch.tensor(gcn.data.y).type(torch.long)

    def test():
        gcn.model.eval()
        # (emb_after_cov1, emb_after_cov2), accs = gcn.model(), []
        (emb_after_cov1, emb_after_cov2), accs = gcn.model(gcn.get_dgl_graph()), []
        logits = F.log_softmax(emb_after_cov2, dim=1)
        # logits,  accs = self.model(), []

        # loss = F.nll_loss(x, y)

        # for _, mask in data('train_mask', 'val_mask', 'test_mask'):
        for _, mask in gcn.data('train_mask', 'val_mask', 'test_mask'): # torch_geometric
            pred = logits[mask].max(1)[1]
            acc = pred.eq(gcn.data.y[mask]).sum().item() / mask.sum().item()
            accs.append(acc)

        # pred = logits.max(1)[1]
        # acc= pred.eq(gcn.data.y).sum().item() / data.y.shape[0]
        # accs.append(acc)

        # for mask in [data.train_mask, data.val_mask, data.test_mask]:
        #     mask = torch.tensor(mask).type(torch.BoolTensor)
        #     pred = logits[mask].max(1)[1]
        #     acc = pred.eq(gcn.data.y[mask]).sum().item() / mask.sum().item()
        #     accs.append(acc)

        return accs

    best_val_acc = test_acc = 0
    performance_history = {}
    for epoch in range(1, 201):
        # embedded_x = train()
        x_after_conv1, x_after_conv2 = gcn.train()
        logits = gcn.discriminator(x_after_conv2)
        gcn.loss_and_step(logits[data.train_mask],
                           data.y[data.train_mask])

        acc = test()

        # log = f'Epoch: {epoch:03d}, acc = {acc}'

        train_acc, val_acc, tmp_test_acc = test()


        if val_acc > best_val_acc:
            best_val_acc = val_acc
            test_acc = tmp_test_acc

        log = f'Epoch: {epoch:03d}, Train: {train_acc:.4f}, Val: {best_val_acc:.4f}, Test: {test_acc:.4f}'
        print(log)

        if epoch / 50 == 0:
            print(f'{epoch}')
            # print(log.format(epoch, train_acc, best_val_acc, test_acc))

