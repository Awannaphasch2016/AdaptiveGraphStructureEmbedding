import os,sys,inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0,parent_dir)

import os.path as osp
from collections import Counter
import scipy.sparse as sp

import numpy as np
import torch
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid

import src.Modeling.gan as  gan_model
import src.Modeling.gcn as gcn_model
from Examples.Models.GAN.utils import Logger
from Log import *



def randomedge_sampler(edge_index , percent):
    """
    Randomly drop edge and preserve percent% edges.
    """

    nnz = edge_index.shape[1]
    perm = np.random.permutation(nnz)
    preserve_nnz = int(nnz * percent)
    perm = perm[:preserve_nnz]
    edge_index[0] = edge_index[0][[perm]]
    edge_index[1] = edge_index[1][[perm]]

    return edge_index


class MyNewModel:
    # def __init__(self, data, dataset, dataloader):
    def __init__(self, data):
        self.data = data
        # self.dataset = dataset
        # self.data_loader = dataloader

        self.init_my_new_model()

    def run_my_new_model(self):
        # TODO where do logger store it to ?

        # best_val_acc = test_acc = 0
        # performance_history = {}

        for epoch in range(self.num_epochs):
            # logger = Logger(model_name='VGAN', data_name='MNIST')

            print(f'======before gcn.train() {epoch}')
            emb_after_conv1, emb_after_conv2 = self.gcn.train()

            # convert edge_index to adj
            data.edge_index = randomedge_sampler(self.data.edge_index, 1)

            labeling_percent = 0.05
            num_labeling = int(data.x.shape[0] * labeling_percent)


           #=====================
           #==
           #=====================

            selected_data_ind = np.random.choice(np.arange(data.x.shape[0]), size=num_labeling, replace=False)
            selected_data_ind_bool = np.zeros(data.x.shape[0])
            selected_data_ind_bool[selected_data_ind] = 1

            self.data.train_mask = torch.tensor(selected_data_ind_bool).type(torch.ByteTensor)
            self.data.test_mask = torch.tensor(np.logical_not(self.data.train_mask.numpy())).type(torch.ByteTensor)

            training_x, training__y = emb_after_conv1[np.where(self.data.train_mask == 1)], self.data.y[np.where(self.data.train_mask == 1)]
            test_x, test__y = emb_after_conv1[np.where(self.data.test_mask == 0)], self.data.y[np.where(self.data.test_mask) == 0]

            #=====================
            #==
            #=====================

            trainning_selected_min_ind = np.intersect1d(np.where(self.data.y == 0), np.where(self.data.train_mask== 1) )
            trainning_selected_maj_ind = np.intersect1d(np.where(self.data.y == 1), np.where(self.data.train_mask== 1) )
            test_selected_min_ind = np.intersect1d(np.where(self.data.y == 0), np.where(self.data.test_mask== 1) )
            test_selected_maj_ind = np.intersect1d(np.where(self.data.y == 1), np.where(self.data.test_mask== 1) )

            trainning_selected_min_y_gcn_emb = self.data.y[trainning_selected_min_ind]
            trainning_selected_min_x_gcn_emb = emb_after_conv1[trainning_selected_min_ind]

            #=====================
            #==gan dataset
            #=====================
            self.min_class_cora_torch_dataset = CoraTorchDataset(trainning_selected_min_x_gcn_emb,
                                                                 trainning_selected_min_y_gcn_emb)

            batch_size = 10
            self.min_class_data_loader_for_gan = torch.utils.data.DataLoader(
                self.min_class_cora_torch_dataset, batch_size=batch_size,
                shuffle=True)


            for n_batch, (real_batch, y) in enumerate(
                    self.min_class_data_loader_for_gan):

                N = real_batch.size(0)
                real_data = real_batch

                fake_data = self.gan.generator(gan_model.noise(N)) # 10, 1433

                d_error, d_pred_real, d_pred_fake = \
                    self.gan.train_discriminator(self.gan.d_optimizer,
                                                 real_data,
                                                 fake_data)

                fake_data = self.gan.generator(gan_model.noise(
                    N))

                g_error = self.gan.train_generator(self.gan.g_optimizer,
                                                   fake_data)
                if n_batch % batch_size == 0:
                    print(f'running GCN epoch = {n_batch}')

            #=====================
            #== fake_neg/true_neg/pos = 25%/25%/50%
            #=====================

            fake_data = self.gan.generator(gan_model.noise(
                N))  # this will be sent to discriminator 2 too
            emb_after_conv1, emb_after_conv2 = self.gcn.train()

            logits = self.gcn.discriminator(emb_after_conv1, fake_data)

            num_select = fake_data.shape[0]

            minreal_minfake_majreal_x = torch.cat((emb_after_conv1, fake_data), 0)
            minreal_minfake_majreal_y = torch.cat((torch.tensor(self.data.y), torch.zeros(fake_data.size(0)).type(torch.int)), 0).type(torch.long)

            trainning_select_min_real_ind = torch.tensor(np.random.choice(trainning_selected_min_ind,size=num_select,replace=False)).type(torch.long)
            trainning_select_maj_real_ind = torch.tensor(np.random.choice(trainning_selected_maj_ind,size=num_select * 2,replace=False)).type(torch.long)
            test_select_min_real_ind = torch.tensor(np.random.choice(test_selected_min_ind,size=num_select,replace=False)).type(torch.long)
            test_select_maj_real_ind = torch.tensor(np.random.choice(test_selected_maj_ind,size=num_select * 2,replace=False)).type(torch.long)
            select_min_fake_ind = torch.tensor(np.arange(fake_data.shape[0]) + self.data.y.shape[0]).type(torch.long)

            trainning_select_minfake_minreal_majreal_ind = torch.cat((trainning_select_min_real_ind, trainning_select_maj_real_ind, select_min_fake_ind), 0)
            test_select_minfake_minreal_majreal_ind = torch.cat((test_select_min_real_ind, test_select_maj_real_ind, select_min_fake_ind), 0)
            # TODO for loss and acc, fake_min 10 (chekc) + train_true_min 10 + train_true_maj 20 (the last two must be selected from training set)
            select_minreal_minfake_majreal_ind_boolean = torch.zeros(minreal_minfake_majreal_x.shape[0])
            select_minreal_minfake_majreal_ind_boolean[trainning_select_minfake_minreal_majreal_ind] = 1
            trainning_select_minreal_minfake_majreal_ind_boolean = select_minreal_minfake_majreal_ind_boolean.type(torch.ByteTensor)

            select_minreal_minfake_majreal_ind_boolean = torch.zeros(minreal_minfake_majreal_x.shape[0])
            select_minreal_minfake_majreal_ind_boolean[test_select_minfake_minreal_majreal_ind] = 1
            test_select_minreal_minfake_majreal_ind_boolean = select_minreal_minfake_majreal_ind_boolean.type(torch.ByteTensor)

            trainning_select_minreal_minfake_majreal_ind_boolean = np.random.permutation(trainning_select_minreal_minfake_majreal_ind_boolean)
            test_select_minreal_minfake_majreal_ind_boolean = np.random.permutation(test_select_minreal_minfake_majreal_ind_boolean)

            trainning_select_minreal_minfake_majreal_ind_boolean = torch.tensor(trainning_select_minreal_minfake_majreal_ind_boolean).type(torch.BoolTensor)
            test_select_minreal_minfake_majreal_ind_boolean = torch.tensor(test_select_minreal_minfake_majreal_ind_boolean).type(torch.BoolTensor)

            self.data.train_mask = trainning_select_minreal_minfake_majreal_ind_boolean
            self.data.test_mask = test_select_minreal_minfake_majreal_ind_boolean

            trainning_select_minreal_minfake_majreal_x =  minreal_minfake_majreal_x[trainning_select_minreal_minfake_majreal_ind_boolean]
            trainning_select_minreal_minfake_majreal_y =  minreal_minfake_majreal_y[trainning_select_minreal_minfake_majreal_ind_boolean]
            test_select_minreal_minfake_majreal_x =  minreal_minfake_majreal_x[test_select_minreal_minfake_majreal_ind_boolean]
            test_select_minreal_minfake_majreal_y =  minreal_minfake_majreal_y[test_select_minreal_minfake_majreal_ind_boolean]


            trainning_loss = self.gcn.loss_and_step(trainning_select_minreal_minfake_majreal_x,trainning_select_minreal_minfake_majreal_y )
            test_loss = self.gcn.loss(test_select_minreal_minfake_majreal_x, test_select_minreal_minfake_majreal_y)

            self.gcn.model.eval()

            (emb_after_cov1, emb_after_cov2), accs = self.gcn.model(self.gcn.get_dgl_graph()), []

            # for test
            logits = self.gcn.discriminator(emb_after_conv1, fake_data)

            for _, mask in data('train_mask', 'test_mask'):
                pred = logits[mask].max(1)[1]
                acc = pred.eq(minreal_minfake_majreal_y[mask]).sum().item() / mask.sum().item()
                accs.append(acc)

            train_acc, tmp_test_acc = accs

            #=====================
            #==gcn performance
            #=====================

            print(f'Epoch: {epoch:03d}, Train: (acc={train_acc:.4f}, loss={trainning_loss:.4f}), Test: (acc={tmp_test_acc:.4f}, loss={test_loss:.4f})')

    def init_my_new_model(self):
        # =====================
        # ==hyper parameters setup
        # =====================
        self.num_batches = 1
        self.num_epochs = 200

        #=====================
        #==relabeling
        #=====================

        new_y = relabel_minority_and_majority_classes(self.data)
        self.data.y = new_y
        self.data.num_classes = np.unique(self.data.y).shape[0]

        # # TODO torch dataset
        # min_class_ind = np.where(self.data.y == 0)
        # min_y_data = self.data.y[min_class_ind]
        # min_x_data = self.data.x[min_class_ind]

        # #=====================
        # #==gan dataset
        # #=====================
        # self.min_class_cora_torch_dataset = CoraTorchDataset(min_x_data,
        #                                                      min_y_data)
        # self.min_class_data_loader_for_gan = torch.utils.data.DataLoader(
        #     self.min_class_cora_torch_dataset, batch_size=100,
        #     shuffle=True)

        # =====================
        # ==for gan
        # =====================
        self.gan = gan_model.GAN(self.data)
        self.gan.init_gan()

        # =====================
        # ==for Gcn
        # ====================
        self.gcn = gcn_model.GCN(self.data)


def ratio_func(y, multiplier, minority_class):
    target_stats = Counter(y)
    return {minority_class: int(multiplier * target_stats[minority_class])}


def relabel_minority_and_majority_classes(data):
    uniq_labels = np.unique(data.y, return_counts=True)
    minority_class = np.unique(data.y, return_counts=True)[1].argmax()
    new_y = np.array([0 if i == minority_class else 1 for i in data.y])
    return new_y

from torch.utils.data import Dataset


class CoraTorchDataset(Dataset):
    def __init__(self, data, y):
        self.data = data
        self.y = y

    def __len__(self):
        return self.data.size(0)

    def __getitem__(self, idx):
        return self.data[idx], self.y[idx]


if __name__ == '__main__':
    # dataset = 'Cora'
    # path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'Data',
    #                 'External')
    #
    # dataset = Planetoid(path, dataset, T.NormalizeFeatures())
    # data = dataset[0]
    # relabel_minority_and_majority_classes(data)

    data, _ = torch.load(r'C:\Users\Anak\PycharmProjects\AdaptiveGraphStructureEmbedding\Notebook\data\Cora\Cora\processed\data.pt')
    new_y = relabel_minority_and_majority_classes(data)
    data.y = new_y
    data.num_classes = np.unique(data.y).shape[0]

    new_y = relabel_minority_and_majority_classes(data)
    data.y = new_y
    data.num_classes = np.unique(data.y).shape[0]

    # todo here>> convert torch geometric data to torch data
    my_new_model = MyNewModel(data)
    my_new_model.run_my_new_model()
