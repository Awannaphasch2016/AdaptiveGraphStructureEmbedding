import os,sys,inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0,parent_dir)
# import warnings
#
# warnings.simplefilter("error")
# warnings.simplefilter("ignore", DeprecationWarning)
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



# def randomedge_sampler(percent, normalization, train_adj):
def randomedge_sampler(edge_index , percent):
    """
    Randomly drop edge and preserve percent% edges.
    """
    "Opt here"

    nnz = edge_index.shape[1]
    perm = np.random.permutation(nnz)
    preserve_nnz = int(nnz * percent)
    perm = perm[:preserve_nnz]
    edge_index[0] = edge_index[0][[perm]]
    edge_index[1] = edge_index[1][[perm]]

    return edge_index


class MyNewModel:
    def __init__(self, data, dataset, dataloader):
        self.data = data
        self.dataset = dataset
        self.data_loader = dataloader

        self.init_my_new_model()

    # def run_gcn_till_converge(self):
    #     self.gcn.train()

    # def run_mlp(self):
    #
    #     # TODO here>> only extract code for mlp part()
    #     logger = Logger(model_name='VGAN', data_name='MNIST')
    #
    #     for n_batch, (real_batch, y) in enumerate(self.data_loader_for_gan):
    #
    #         N = real_batch.size(0)
    #         real_data = real_batch
    #         ### for torch_geometric dataloader
    #         # batch_ind = batch_ind[1]
    #         # real_data = x[1]
    #         # N = real_data.size(0)
    #
    #         fake_data = self.gan.generator(gan_model.noise(N)).detach()
    #         # return fake_data
    #         d_error, d_pred_real, d_pred_fake = \
    #             self.gan.train_discriminator(self.gan.d_optimizer, real_data,
    #                                          fake_data)
    #
    #         # 2. Train Generator
    #         # Generate fake data
    #         fake_data = self.gan.generator(gan_model.noise(
    #             N))  # this will be sent to discriminator 2 too
    #
    #         # # TODO
    #         # self.run_gcn_till_converge()
    #
    #         # Train G
    #         g_error = self.gan.train_generator(self.gan.g_optimizer, fake_data)
    #         # Log batch error
    #         logger.log(d_error, g_error, epoch, n_batch, self.num_batches)
    #         # Display Progress every few batches
    #         if (n_batch) % 100 == 0:
    #             # Display status Logs
    #             logger.display_status(
    #                 epoch, self.num_epochs, n_batch, self.num_batches,
    #                 d_error, g_error, d_pred_real, d_pred_fake)

    def run_my_new_model(self):
        # TODO where do logger store it to ?

        best_val_acc = test_acc = 0
        performance_history = {}

        for epoch in range(self.num_epochs):
            logger = Logger(model_name='VGAN', data_name='MNIST')

            # TODO how to program with 2 separate graph

            print(f'======before gcn.train() {epoch}')
            emb_after_conv1, emb_after_conv2 = self.gcn.train()
            # TODO how to get adj from data.  (Cora) dataset

            # TODO here>> randomly label 10 percent to 20 percent labels ( before runnign the result)
            # TODO run gcn with minority and majority class ( get bench mark result)
            # convert edge_index to adj
            data.edge_index = randomedge_sampler(self.data.edge_index, 1)
            # todo change self.data.y to selected y ( same goes for x)
            labeling_percent = 0.05
            num_labeling = int(data.x.shape[0] * labeling_percent)


            # TODO add train and test set ( in data )
            selected_data_ind = np.random.choice(np.arange(data.x.shape[0]), size=num_labeling, replace=False)
            selected_data_ind_bool = np.zeros(data.x.shape[0])
            selected_data_ind_bool[selected_data_ind] = 1
            # selected_emb_x, selected_y = emb_after_conv1[np.where(selected_data_ind == 1)], data.y[np.where(selected_data_ind == 1)]
            self.data.train_mask = torch.tensor(selected_data_ind_bool).type(torch.ByteTensor)
            self.data.test_mask = torch.tensor(np.logical_not(self.data.train_mask.numpy())).type(torch.ByteTensor)
            # self.data.test_mask = ~self.data.train_mask.type(torch.ByteTensor)
            training_x, training__y = emb_after_conv1[np.where(self.data.train_mask == 1)], self.data.y[np.where(self.data.train_mask == 1)]
            test_x, test__y = emb_after_conv1[np.where(self.data.test_mask == 0)], self.data.y[np.where(self.data.test_mask) == 0]

            # min_y_gcn_emb = torch.tensor(self.data.y[np.where(self.data.y == 0)])
            # min_x_gcn_emb = emb_after_conv1[np.where(self.data.y == 0)] # are labels ordered by order of x?
            # TODO select minority from selected 20 percent.
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

            # n_batch = 1
            # real_batch = next(iter(self.min_class_data_loader_for_gan))[0]

            for n_batch, (real_batch, y) in enumerate(
                    self.min_class_data_loader_for_gan):

            # if True:
                # TODO here>> why does real have feature = 16 and fake have feature = 1433??
                N = real_batch.size(0)
                real_data = real_batch
                ### for torch_geometric dataloader
                # batch_ind = batch_ind[1]
                # real_data = x[1]
                # N = real_data.size(0)

                # gen_labels = self.gan.get_gen_labels_for_real_fake_minority_class(N)
                # fake_data = self.gan.generator(gan_model.noise(N)).detach()
                fake_data = self.gan.generator(gan_model.noise(N)) # 10, 1433

                # print(f'before train discriminator {n_batch}')
                d_error, d_pred_real, d_pred_fake = \
                    self.gan.train_discriminator(self.gan.d_optimizer,
                                                 real_data,
                                                 fake_data)

                fake_data = self.gan.generator(gan_model.noise(
                    N))  # this will be sent to discriminator 2 too

                # Train G
                # print(f'before train generator {n_batch}')
                g_error = self.gan.train_generator(self.gan.g_optimizer,
                                                   fake_data)
                if n_batch % batch_size == 0:
                    print(f'running GCN epoch = {n_batch}')

            # TODO here>> run discriminator of gcn where input is fake_neg ( 25%), true_neg(25$), pos (50%)
            # TODO here>> where is maj and min labeled are used in loss funciton

            fake_data = self.gan.generator(gan_model.noise(
                N))  # this will be sent to discriminator 2 too
            emb_after_conv1, emb_after_conv2 = self.gcn.train()

            # TODO logits should only have 2 dim. how shoudl I reduce 16 dim to 2 ( do it in GAN? or do it in GCN?)
            logits = self.gcn.discriminator(emb_after_conv1, fake_data)

            num_select = fake_data.shape[0]

            # TODO change gcn label rate
            # minreal_minfake_majreal_size = np.arange(training_gcn_real_x.shape[0] + fake_data.shape[0]).shape[0]
            minreal_minfake_majreal_x = torch.cat((emb_after_conv1, fake_data), 0)
            minreal_minfake_majreal_y = torch.cat((torch.tensor(self.data.y), torch.zeros(fake_data.size(0)).type(torch.int)), 0).type(torch.long)

            # trainning_minreal_minfake_majreal_x = minreal_minfake_majreal_x[self.data.train_mask]
            # trainning_minreal_minfake_majreal_y = minreal_minfake_majreal_y[self.data.train_mask]
            # test_minreal_minfake_majreal_x = minreal_minfake_majreal_x[self.data.test_mask]
            # test_minreal_minfake_majreal_y = minreal_minfake_majreal_y[self.data.test_mask]


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

            # self.data.y = minreal_minfake_majreal_y
            trainning_select_minreal_minfake_majreal_ind_boolean = torch.tensor(trainning_select_minreal_minfake_majreal_ind_boolean).type(torch.BoolTensor)
            test_select_minreal_minfake_majreal_ind_boolean = torch.tensor(test_select_minreal_minfake_majreal_ind_boolean).type(torch.BoolTensor)

            self.data.train_mask = trainning_select_minreal_minfake_majreal_ind_boolean
            self.data.test_mask = test_select_minreal_minfake_majreal_ind_boolean

            trainning_select_minreal_minfake_majreal_x =  minreal_minfake_majreal_x[trainning_select_minreal_minfake_majreal_ind_boolean]
            trainning_select_minreal_minfake_majreal_y =  minreal_minfake_majreal_y[trainning_select_minreal_minfake_majreal_ind_boolean]
            test_select_minreal_minfake_majreal_x =  minreal_minfake_majreal_x[test_select_minreal_minfake_majreal_ind_boolean]
            test_select_minreal_minfake_majreal_y =  minreal_minfake_majreal_y[test_select_minreal_minfake_majreal_ind_boolean]


            # for trainn set
            trainning_loss = self.gcn.loss_and_step(trainning_select_minreal_minfake_majreal_x,trainning_select_minreal_minfake_majreal_y )
            test_loss = self.gcn.loss(test_select_minreal_minfake_majreal_x, test_select_minreal_minfake_majreal_y)

            # # for trainning
            # logits = self.gcn.discriminator(emb_after_conv1, fake_data)

            self.gcn.model.eval()

            (emb_after_cov1, emb_after_cov2), accs = self.gcn.model(self.gcn.get_dgl_graph()), []
            # for test
            logits = self.gcn.discriminator(emb_after_conv1, fake_data)
            # logits = F.log_softmax(emb_after_cov2, dim=1)
            # logits,  accs = self.model(), []

            # loss = F.nll_loss(x, y)
            # for _, mask in data('train_mask', 'val_mask', 'test_mask'):

            for _, mask in data('train_mask', 'test_mask'):
                pred = logits[mask].max(1)[1]
                acc = pred.eq(minreal_minfake_majreal_y[mask]).sum().item() / mask.sum().item()
                accs.append(acc)

            train_acc, tmp_test_acc = accs

            # self.gcn.test()
            # self.gcn.run_gcn()
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
        self.gan = gan_model.GAN(self.data, self.data_loader)
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
    dataset = 'Cora'
    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'Data',
                    'External')

    dataset = Planetoid(path, dataset, T.NormalizeFeatures())
    data = dataset[0]
    # relabel_minority_and_majority_classes(data)

    from torch_geometric.data import DataLoader

    dataloader = DataLoader(dataset)  # not re-labeled

    # todo here>> convert torch geometric data to torch data
    my_new_model = MyNewModel(data, dataset, dataloader)
    my_new_model.run_my_new_model()
