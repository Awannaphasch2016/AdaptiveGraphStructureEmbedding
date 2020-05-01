import inspect
import os
import sys
import time

current_dir = os.path.dirname(
    os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from collections import Counter

import numpy as np
import torch

import src.Modeling.gan as  gan_model
import src.Modeling.gcn as gcn_model
import Log.Logger as Logging
from arg_parser import args
from src.Preparation.Data import preparing_cora_for_new_purposed_model
# from Plot import PlotClass
from src.Visualization import PlotClass
from src.Evaluation import get_total_roc_auc_score

log = Logging.Logger(name='log_for_train_model_file')


def randomedge_sampler(edge_index, percent, isLog=True):
    """
    Randomly drop edge and preserve percent% edges.
    """
    if isLog:
        log.info('in randomedge_sampler..')

    nnz = edge_index.shape[1]
    perm = np.random.permutation(nnz)
    preserve_nnz = int(nnz * percent)
    perm = perm[:preserve_nnz]
    edge_index[0] = edge_index[0][[perm]]
    edge_index[1] = edge_index[1][[perm]]

    return edge_index


class ModelInputData():
    def __init__(self, data):
        self.data = data

    def prepare_ind_for_trainning_and_test_set(self):
        labeling_percent = 0.05
        num_labeling = int(self.data.x.shape[0] * labeling_percent)

        # =====================
        # ==control for probability for each class
        # =====================

        self.label_count = np.unique(self.data.y_before_relabel,
                                     return_counts=True)
        self.label_count_dict = {i: 1 / self.data.y.shape[0] for (i, j) in
                                 zip(self.label_count[0], self.label_count[1])}
        self.p = np.array([self.label_count_dict[i] for i in
                           self.data.y_before_relabel.numpy()])
        self.p /= self.p.sum()
        selected_data_ind = np.random.choice(np.arange(self.data.x.shape[0]),
                                             size=num_labeling, replace=False,
                                             p=self.p)

        selected_data_ind_bool = np.zeros(self.data.x.shape[0])
        selected_data_ind_bool[selected_data_ind] = 1

        self.data.train_mask = torch.tensor(selected_data_ind_bool).type(
            torch.ByteTensor)
        self.data.test_mask = torch.tensor(
            np.logical_not(self.data.train_mask.numpy())).type(torch.ByteTensor)

        trainning_selected_min_ind = np.intersect1d(np.where(self.data.y == 0),
                                                    np.where(
                                                        self.data.train_mask == 1))
        trainning_selected_maj_ind = np.intersect1d(np.where(self.data.y == 1),
                                                    np.where(
                                                        self.data.train_mask == 1))
        test_selected_min_ind = np.intersect1d(np.where(self.data.y == 0),
                                               np.where(
                                                   self.data.test_mask == 1))
        test_selected_maj_ind = np.intersect1d(np.where(self.data.y == 1),
                                               np.where(
                                                   self.data.test_mask == 1))
        return (trainning_selected_min_ind, trainning_selected_maj_ind), (
            test_selected_min_ind, test_selected_maj_ind)

    def set_data(self, minreal_minfake_majreal_x, minreal_minfake_majreal_y,
                 emb_after_conv1, fake_data, num_select,
                 trainning_selected_min_ind, trainning_selected_maj_ind,
                 test_selected_min_ind, test_selected_maj_ind):
        self.fake_data = fake_data
        self.emb_after_conv1 = emb_after_conv1
        self.num_select = num_select
        self.trainning_selected_min_ind = trainning_selected_min_ind
        self.trainning_selected_maj_ind = trainning_selected_maj_ind
        self.test_selected_min_ind = test_selected_min_ind
        self.test_selected_maj_ind = test_selected_maj_ind
        self.minreal_minfake_majreal_x = minreal_minfake_majreal_x
        self.minreal_minfake_majreal_y = minreal_minfake_majreal_y

    def set_train_test_data_index(self):
        # minreal_minfake_majreal_x = torch.cat(
        #     (self.emb_after_conv1, self.fake_data), 0)
        # minreal_minfake_majreal_y = torch.cat((torch.tensor(self.data.y),
        #                                        torch.zeros(
        #                                            self.fake_data.size(0)).type(
        #                                            torch.int)), 0).type(
        #     torch.long)

        self.prepare_gcn_dataset(
            self.num_select, self.minreal_minfake_majreal_x,
            self.minreal_minfake_majreal_y, self.fake_data,
            self.trainning_selected_min_ind,
            self.trainning_selected_maj_ind,
            self.test_selected_min_ind,
            self.test_selected_maj_ind
        )

    def prepare_gcn_dataset(self, num_select, minreal_minfake_majreal_x,
                            minreal_minfake_majreal_y, fake_data,
                            trainning_selected_min_ind,
                            trainning_selected_maj_ind,
                            test_selected_min_ind,
                            test_selected_maj_ind):
        trainning_select_min_real_ind = torch.tensor(
            np.random.choice(trainning_selected_min_ind, size=num_select,
                             replace=False)).type(torch.long)
        trainning_select_maj_real_ind = torch.tensor(
            np.random.choice(trainning_selected_maj_ind, size=num_select * 2,
                             replace=False)).type(torch.long)
        test_select_min_real_ind = torch.tensor(
            np.random.choice(test_selected_min_ind, size=num_select,
                             replace=False)).type(torch.long)
        test_select_maj_real_ind = torch.tensor(
            np.random.choice(test_selected_maj_ind, size=num_select * 2,
                             replace=False)).type(torch.long)

        select_min_fake_ind = torch.tensor(
            np.arange(fake_data.shape[0]) + self.data.y.shape[0]).type(
            torch.long)


        trainning_select_minfake_minreal_majreal_ind = torch.cat((
            trainning_select_min_real_ind,
            trainning_select_maj_real_ind,
            select_min_fake_ind),
            0)

        test_select_minfake_minreal_majreal_ind = torch.cat((
            test_select_min_real_ind,
            test_select_maj_real_ind,
            select_min_fake_ind),
            0)

        select_minreal_minfake_majreal_ind_boolean = torch.zeros(
            minreal_minfake_majreal_x.shape[0])
        select_minreal_minfake_majreal_ind_boolean[
            trainning_select_minfake_minreal_majreal_ind] = 1
        trainning_select_minreal_minfake_majreal_ind_boolean = select_minreal_minfake_majreal_ind_boolean.type(
            torch.ByteTensor)
        #=====================
        #==min_fake, min_real, maj boolean ind
        #=====================


        trainning_select_min_real_boolean = torch.zeros(
            minreal_minfake_majreal_x.shape[0])
        trainning_select_min_real_boolean[trainning_select_min_real_ind] = 1

        trainning_select_maj_real_boolean = torch.zeros(
            minreal_minfake_majreal_x.shape[0])
        trainning_select_maj_real_boolean[trainning_select_maj_real_ind] = 1

        test_select_min_real_boolean = torch.zeros(
            minreal_minfake_majreal_x.shape[0])
        test_select_min_real_boolean[test_select_min_real_ind] = 1

        test_select_maj_real_boolean = torch.zeros(
            minreal_minfake_majreal_x.shape[0])
        test_select_maj_real_boolean[test_select_maj_real_ind] = 1

        select_min_fake_boolean = torch.zeros(
            minreal_minfake_majreal_x.shape[0])
        select_min_fake_boolean[select_min_fake_ind] = 1

        self.trainning_select_min_real_boolean = trainning_select_min_real_boolean.type(torch.BoolTensor)
        self.trainning_select_maj_real_boolean = trainning_select_maj_real_boolean.type(torch.BoolTensor)
        self.test_select_min_real_boolean = test_select_min_real_boolean.type(torch.BoolTensor)
        self.test_select_maj_real_boolean = test_select_maj_real_boolean.type(torch.BoolTensor)
        self.select_min_fake_boolean = select_min_fake_boolean.type(torch.BoolTensor)

        #=====================
        #==trainning_select_minreal_minfake_majreal_ind_boolean, test_select_minreal_minfake_majreal_ind_boolean
        #=====================
         
        select_minreal_minfake_majreal_ind_boolean = torch.zeros(
            minreal_minfake_majreal_x.shape[0])
        select_minreal_minfake_majreal_ind_boolean[
            test_select_minfake_minreal_majreal_ind] = 1
        test_select_minreal_minfake_majreal_ind_boolean = select_minreal_minfake_majreal_ind_boolean.type(
            torch.ByteTensor)

        trainning_select_minreal_minfake_majreal_ind_boolean = np.random.permutation(
            trainning_select_minreal_minfake_majreal_ind_boolean)
        test_select_minreal_minfake_majreal_ind_boolean = np.random.permutation(
            test_select_minreal_minfake_majreal_ind_boolean)

        trainning_select_minreal_minfake_majreal_ind_boolean = torch.tensor(
            trainning_select_minreal_minfake_majreal_ind_boolean).type(
            torch.BoolTensor)
        test_select_minreal_minfake_majreal_ind_boolean = torch.tensor(
            test_select_minreal_minfake_majreal_ind_boolean).type(
            torch.BoolTensor)

        self.data.train_mask = trainning_select_minreal_minfake_majreal_ind_boolean
        self.data.test_mask = test_select_minreal_minfake_majreal_ind_boolean

        trainning_select_minreal_minfake_majreal_x = minreal_minfake_majreal_x[
            trainning_select_minreal_minfake_majreal_ind_boolean]
        trainning_select_minreal_minfake_majreal_y = minreal_minfake_majreal_y[
            trainning_select_minreal_minfake_majreal_ind_boolean]
        test_select_minreal_minfake_majreal_x = minreal_minfake_majreal_x[
            test_select_minreal_minfake_majreal_ind_boolean]
        test_select_minreal_minfake_majreal_y = minreal_minfake_majreal_y[
            test_select_minreal_minfake_majreal_ind_boolean]

        self.trainning_select_minreal_minfake_majreal_ind_boolean = trainning_select_minreal_minfake_majreal_ind_boolean
        self.test_select_minreal_minfake_majreal_ind_boolean = test_select_minreal_minfake_majreal_ind_boolean
        self.trainning_select_minreal_minfake_majreal_y = trainning_select_minreal_minfake_majreal_y
        self.test_select_minreal_minfake_majreal_y = test_select_minreal_minfake_majreal_y


class MyNewModel:
    # def __init__(self, data, dataset, dataloader):
    def __init__(self, data, isLog=False):
        self.scan_fakemin_realmin_maj_performance = True
        self.time_stamp = time.strftime("%Y%m%d-%H%M%S")

        self.plot_class = PlotClass()
        # self.plot_class.set_subplots((2, 1))
        if self.scan_fakemin_realmin_maj_performance:
            self.plot_class.set_subplots((3, 1))
        else:
            self.plot_class.set_subplots((3, 1))

        self.data = data
        self.log = isLog

        self.init_my_new_model()
        self.model_input_data = ModelInputData(data)

    def prepare_gan_trainning_dataset(self, emb_after_conv1,
                                      trainning_selected_min_ind):
        """There are only minority samples that is used as input"""

        trainning_selected_min_y_gcn_emb = self.data.y[
            trainning_selected_min_ind]
        trainning_selected_min_x_gcn_emb = emb_after_conv1[
            trainning_selected_min_ind]

        self.min_class_cora_torch_dataset = CoraTorchDataset(
            trainning_selected_min_x_gcn_emb,
            trainning_selected_min_y_gcn_emb)

        self.batch_size = 10

        return torch.utils.data.DataLoader(
            self.min_class_cora_torch_dataset, batch_size=self.batch_size,
            shuffle=True)

    def prepare_ind_for_trainning_and_test_set(self):
        if self.log:
            log.info('in prepare_ind_for_trainning_and_test_set..')

        labeling_percent = 0.05
        num_labeling = int(data.x.shape[0] * labeling_percent)

        selected_data_ind = np.random.choice(np.arange(data.x.shape[0]),
                                             size=num_labeling, replace=False)
        selected_data_ind_bool = np.zeros(data.x.shape[0])
        selected_data_ind_bool[selected_data_ind] = 1

        self.data.train_mask = torch.tensor(selected_data_ind_bool).type(
            torch.ByteTensor)
        self.data.test_mask = torch.tensor(
            np.logical_not(self.data.train_mask.numpy())).type(torch.ByteTensor)

        trainning_selected_min_ind = np.intersect1d(np.where(self.data.y == 0),
                                                    np.where(
                                                        self.data.train_mask == 1))
        trainning_selected_maj_ind = np.intersect1d(np.where(self.data.y == 1),
                                                    np.where(
                                                        self.data.train_mask == 1))
        test_selected_min_ind = np.intersect1d(np.where(self.data.y == 0),
                                               np.where(
                                                   self.data.test_mask == 1))
        test_selected_maj_ind = np.intersect1d(np.where(self.data.y == 1),
                                               np.where(
                                                   self.data.test_mask == 1))
        return (trainning_selected_min_ind, trainning_selected_maj_ind), (
            test_selected_min_ind, test_selected_maj_ind)

    def run_gan_components_of_new_model(self):
        if self.log:
            log.info('in run_gan_components_of_new_model...')
        for n_batch, (real_batch, y) in enumerate(
                self.min_class_data_loader_for_gan):
            self.number_of_sample_per_batch = real_batch.size(0)
            real_data = real_batch

            fake_data = self.gan.generator(
                gan_model.noise(self.number_of_sample_per_batch))  # 10, 1433

            d_error, d_pred_real, d_pred_fake = \
                self.gan.train_discriminator(self.gan.d_optimizer,
                                             real_data,
                                             fake_data)

            fake_data = self.gan.generator(gan_model.noise(
                self.number_of_sample_per_batch))

            g_error = self.gan.train_generator(self.gan.g_optimizer,
                                               fake_data)
            if self.log:
                log.info(f'running GCN epoch = {n_batch}')

    def prepare_gcn_dataset(self, num_select, minreal_minfake_majreal_x,
                            minreal_minfake_majreal_y, fake_data,
                            trainning_selected_min_ind,
                            trainning_selected_maj_ind,
                            test_selected_min_ind,
                            test_selected_maj_ind):

        trainning_select_min_real_ind = torch.tensor(
            np.random.choice(trainning_selected_min_ind, size=num_select,
                             replace=False)).type(torch.long)
        trainning_select_maj_real_ind = torch.tensor(
            np.random.choice(trainning_selected_maj_ind, size=num_select * 2,
                             replace=False)).type(torch.long)
        test_select_min_real_ind = torch.tensor(
            np.random.choice(test_selected_min_ind, size=num_select,
                             replace=False)).type(torch.long)
        test_select_maj_real_ind = torch.tensor(
            np.random.choice(test_selected_maj_ind, size=num_select * 2,
                             replace=False)).type(torch.long)

        select_min_fake_ind = torch.tensor(
            np.arange(fake_data.shape[0]) + self.data.y.shape[0]).type(
            torch.long)

        trainning_select_minfake_minreal_majreal_ind = torch.cat((
            trainning_select_min_real_ind,
            trainning_select_maj_real_ind,
            select_min_fake_ind),
            0)

        test_select_minfake_minreal_majreal_ind = torch.cat((
            test_select_min_real_ind,
            test_select_maj_real_ind,
            select_min_fake_ind),
            0)

        select_minreal_minfake_majreal_ind_boolean = torch.zeros(
            minreal_minfake_majreal_x.shape[0])
        select_minreal_minfake_majreal_ind_boolean[
            trainning_select_minfake_minreal_majreal_ind] = 1
        trainning_select_minreal_minfake_majreal_ind_boolean = select_minreal_minfake_majreal_ind_boolean.type(
            torch.ByteTensor)

        select_minreal_minfake_majreal_ind_boolean = torch.zeros(
            minreal_minfake_majreal_x.shape[0])
        select_minreal_minfake_majreal_ind_boolean[
            test_select_minfake_minreal_majreal_ind] = 1
        test_select_minreal_minfake_majreal_ind_boolean = select_minreal_minfake_majreal_ind_boolean.type(
            torch.ByteTensor)

        trainning_select_minreal_minfake_majreal_ind_boolean = np.random.permutation(
            trainning_select_minreal_minfake_majreal_ind_boolean)
        test_select_minreal_minfake_majreal_ind_boolean = np.random.permutation(
            test_select_minreal_minfake_majreal_ind_boolean)

        trainning_select_minreal_minfake_majreal_ind_boolean = torch.tensor(
            trainning_select_minreal_minfake_majreal_ind_boolean).type(
            torch.BoolTensor)
        test_select_minreal_minfake_majreal_ind_boolean = torch.tensor(
            test_select_minreal_minfake_majreal_ind_boolean).type(
            torch.BoolTensor)

        self.data.train_mask = trainning_select_minreal_minfake_majreal_ind_boolean
        self.data.test_mask = test_select_minreal_minfake_majreal_ind_boolean

        trainning_select_minreal_minfake_majreal_x = minreal_minfake_majreal_x[
            trainning_select_minreal_minfake_majreal_ind_boolean]
        trainning_select_minreal_minfake_majreal_y = minreal_minfake_majreal_y[
            trainning_select_minreal_minfake_majreal_ind_boolean]
        test_select_minreal_minfake_majreal_x = minreal_minfake_majreal_x[
            test_select_minreal_minfake_majreal_ind_boolean]
        test_select_minreal_minfake_majreal_y = minreal_minfake_majreal_y[
            test_select_minreal_minfake_majreal_ind_boolean]

        return (
                   trainning_select_minreal_minfake_majreal_ind_boolean,
                   test_select_minreal_minfake_majreal_ind_boolean), (
                   trainning_select_minreal_minfake_majreal_x,
                   trainning_select_minreal_minfake_majreal_y), (
                   test_select_minreal_minfake_majreal_x,
                   test_select_minreal_minfake_majreal_y)

    def collect_performance(self, logits, minreal_minfake_majreal_y,
                            scan_fakemin_realmin_maj_performance):

        y_pred_dict = {}
        y_score_dict = {}
        y_true_dict = {}

        accs_dict = {}
        aucs_dict = {}

        # if scan_fakemin_realmin_maj_performance:

        for name, mask in self.data('train_mask', 'test_mask',
                                    'trainning_select_min_real_boolean',
                                    'trainning_select_maj_real_boolean',
                                    'test_select_min_real_boolean',
                                    'test_select_maj_real_boolean',
                                    'select_min_fake_boolean'):
            pred = logits[mask].max(1)[1]
            y_true = minreal_minfake_majreal_y[mask]
            y_score = logits[mask].detach().numpy()


            y_pred_dict.setdefault(name, pred.detach().numpy())
            y_score_dict.setdefault(name, y_score)
            y_true_dict.setdefault(name, y_true.detach().numpy())

            acc = pred.eq(minreal_minfake_majreal_y[
                              mask]).sum().item() / mask.sum().item()
            accs_dict.setdefault(name, acc)

            if name in ['train_mask', 'test_mask']:
                auc = get_total_roc_auc_score(y_true, y_score)
                aucs_dict.setdefault(name, auc)

        # else:
        #     # TODO what is the index of fake
        #     for name, mask in self.data('train_mask', 'test_mask'):
        #         pred = logits[mask].max(1)[1]
        #         y_true = minreal_minfake_majreal_y[mask]
        #         y_score = logits[mask].detach().numpy()
        #         auc = get_total_roc_auc_score(y_true, y_score)
        #
        #         y_pred_dict.setdefault(name, pred.detach().numpy())
        #         y_score_dict.setdefault(name, y_score)
        #         y_true_dict.setdefault(name, y_true.detach().numpy())
        #         auc_dict.setdefault(name, auc)
        #
        #         acc = pred.eq(minreal_minfake_majreal_y[
        #                           mask]).sum().item() / mask.sum().item()
        #         accs.append(acc)
        #         aucs.append(auc)

        return accs_dict, aucs_dict, y_pred_dict, y_score_dict, y_true_dict

    def run_my_new_model_once(self, epoch,
                              scan_fakemin_realmin_maj_performance=False):
        if self.log:
            log.info('in run_my_new_model_once..')

        # emb_after_conv1, emb_after_conv2 = self.gcn.train()
        self.gcn.model.train()

        emb_after_conv1 = self.gcn.model(self.gcn.get_dgl_graph(), self.data.x,
                                         get_conv1_emb=True)

        # convert edge_index to adj
        # TODO does this have any connection to otehr part of the model? maybe I just forget to connect it to other components
        self.data.edge_index = randomedge_sampler(self.data.edge_index, 1,
                                                  isLog=self.log)

        # =====================
        # == select 5 percent for trainning dataset
        # =====================

        # (trainning_selected_min_ind, trainning_selected_maj_ind), (
        #     test_selected_min_ind,
        #     test_selected_maj_ind) = self.prepare_ind_for_trainning_and_test_set()

        # =====================
        # ==gan dataset
        # =====================
        self.min_class_data_loader_for_gan = self.prepare_gan_trainning_dataset(
            emb_after_conv1,
            self.trainning_selected_min_ind)

        self.run_gan_components_of_new_model()

        # =====================
        # == fake_neg/true_neg/pos = 25%/25%/50%
        # =====================

        fake_data = self.gan.generator(gan_model.noise(
            self.number_of_sample_per_batch))  # this will be sent to discriminator 2 too

        num_select = fake_data.shape[0]

        minreal_minfake_majreal_x = torch.cat(
            (emb_after_conv1, fake_data), 0)
        minreal_minfake_majreal_y = torch.cat((torch.tensor(self.data.y),
                                               torch.zeros(
                                                   fake_data.size(0)).type(
                                                   torch.int)), 0).type(
            torch.long)

        if epoch == 0:
            self.model_input_data.set_data(minreal_minfake_majreal_x,
                                           minreal_minfake_majreal_y,
                                           emb_after_conv1,
                                           fake_data, num_select,
                                           self.trainning_selected_min_ind,
                                           self.trainning_selected_maj_ind,
                                           self.test_selected_min_ind,
                                           self.test_selected_maj_ind)
            self.model_input_data.set_train_test_data_index()

        self.data.trainning_select_min_real_boolean = self.model_input_data.trainning_select_min_real_boolean
        self.data.trainning_select_maj_real_boolean = self.model_input_data.trainning_select_maj_real_boolean
        self.data.test_select_min_real_boolean = self.model_input_data.test_select_min_real_boolean
        self.data.test_select_maj_real_boolean = self.model_input_data.test_select_maj_real_boolean
        self.data.select_min_fake_boolean = self.model_input_data.select_min_fake_boolean

        trainning_select_minreal_minfake_majreal_ind_boolean = self.model_input_data.trainning_select_minreal_minfake_majreal_ind_boolean
        test_select_minreal_minfake_majreal_ind_boolean = self.model_input_data.test_select_minreal_minfake_majreal_ind_boolean
        trainning_select_minreal_minfake_majreal_y = self.model_input_data.trainning_select_minreal_minfake_majreal_y
        test_select_minreal_minfake_majreal_y = self.model_input_data.test_select_minreal_minfake_majreal_y

        # emb_after_conv2, logits = self.gcn.model(self.gcn.get_dgl_graph(), self.data.x, external_input=trainning_select_minreal_minfake_majreal_x )
        emb_after_conv2, logits = self.gcn.model(self.gcn.get_dgl_graph(),
                                                 emb_after_conv1,
                                                 external_input=fake_data)

        trainning_loss = self.gcn.loss_and_step(
            logits[trainning_select_minreal_minfake_majreal_ind_boolean],
            trainning_select_minreal_minfake_majreal_y)

        # =====================
        # == gan test
        # =====================
        self.gcn.model.eval()

        (emb_after_conv2, logits), accs, aucs = self.gcn.model(
            self.gcn.get_dgl_graph(), emb_after_conv1,
            external_input=fake_data), [], []
        test_loss = self.gcn.loss(
            logits[test_select_minreal_minfake_majreal_ind_boolean],
            test_select_minreal_minfake_majreal_y)

        # (emb_after_conv1, emb_after_conv2, logits), accs = self.gcn.model(
        #     self.gcn.get_dgl_graph(), self.data.x), []
        # # for test
        # # TODO get logits from gcn classifier aka discriminator
        # logits = self.gcn.discriminator(emb_after_conv1, fake_data)
        # test_loss = self.gcn.loss(test_select_minreal_minfake_majreal_x,
        #                           test_select_minreal_minfake_majreal_y)

        accs_dict, aucs_dict, y_pred_dict, y_score_dict, y_true_dict = self.collect_performance(
            logits, minreal_minfake_majreal_y,
            scan_fakemin_realmin_maj_performance)

        # (trainning_select_min_real_acc,
        #  trainning_select_maj_real_acc,
        #  test_select_min_real_acc,
        #  test_select_maj_real_acc,
        #  select_min_fake_acc) = accs
        # (trainning_select_min_real_auc,
        #  trainning_select_maj_real_auc,
        #  test_select_min_real_auc,
        #  test_select_maj_real_auc,
        #  select_min_fake_auc) = aucs

        if scan_fakemin_realmin_maj_performance:
            print(f"""
                Epoch: {epoch:03d},
                Trainning: (loss: {trainning_loss:.4f}, min_fake: acc={accs_dict["select_min_fake_boolean"]:.4f} min_real: acc={accs_dict["trainning_select_min_real_boolean"]:.4f} maj: acc={accs_dict["trainning_select_maj_real_boolean"]:.4f})
                Test     : (loss: {test_loss:.4f} min_fake: acc={accs_dict["select_min_fake_boolean"]:.4f} min_real: acc={accs_dict["test_select_min_real_boolean"]:.4f} maj: acc={accs_dict["test_select_maj_real_boolean"]:.4f}) """)
            
            self.plot_class.collect_hist('train_loss', trainning_loss)
            self.plot_class.collect_hist('test_loss', test_loss)
            self.plot_class.collect_hist('select_min_fake_boolean', accs_dict['select_min_fake_boolean'])
            self.plot_class.collect_hist('trainning_select_min_real_boolean', accs_dict['trainning_select_min_real_boolean'])
            self.plot_class.collect_hist('trainning_select_maj_real_boolean', accs_dict['trainning_select_maj_real_boolean'])
            self.plot_class.collect_hist('test_select_min_real_boolean', accs_dict['test_select_min_real_boolean'])
            self.plot_class.collect_hist('test_select_maj_real_boolean', accs_dict['test_select_maj_real_boolean'])

        else:
            print(f' Epoch: {epoch:03d},\n'
                  f'Train: (acc={accs_dict["train_mask"]:.4f}, auc={aucs_dict["train_mask"]: .4f}, loss={trainning_loss:.4f}),\n'
                  f'Test: ({accs_dict["test_mask"]:.4f}, auc={aucs_dict["test_mask"]:4f}, loss={test_loss:.4f})')

            self.plot_class.collect_hist('train_loss', trainning_loss)
            self.plot_class.collect_hist('test_loss', test_loss)
            self.plot_class.collect_hist('train_acc', accs_dict['train_mask'])
            self.plot_class.collect_hist('test_acc', accs_dict['test_mask'])
            self.plot_class.collect_hist('train_auc', aucs_dict['train_mask'])
            self.plot_class.collect_hist('test_auc', aucs_dict['test_mask'])

        return y_true_dict, y_pred_dict, y_score_dict, aucs_dict

    def run_my_new_model(self):


        (self.trainning_selected_min_ind, self.trainning_selected_maj_ind), (
            self.test_selected_min_ind,
            self.test_selected_maj_ind) = self.model_input_data.prepare_ind_for_trainning_and_test_set()

        # for epoch in range(self.num_epochs):
        for epoch in range(10):
            # for epoch in range(50):
            y_true_dict, y_pred_dict, y_score_dict, aucs_dict = self.run_my_new_model_once(
                epoch, scan_fakemin_realmin_maj_performance=self.scan_fakemin_realmin_maj_performance)

        if self.scan_fakemin_realmin_maj_performance:
            self.plot_class.plot_each_hist((0, 0), name='train_loss')
            self.plot_class.plot_each_hist((0, 0), name='test_loss')
            self.plot_class.plot_each_hist((1, 0), name='select_min_fake_boolean')
            self.plot_class.plot_each_hist((1, 0), name='trainning_select_min_real_boolean')
            self.plot_class.plot_each_hist((1, 0), name='trainning_select_maj_real_boolean')
            self.plot_class.plot_each_hist((2, 0), name='select_min_fake_boolean')
            self.plot_class.plot_each_hist((2, 0), name='test_select_min_real_boolean')
            self.plot_class.plot_each_hist((2, 0), name='test_select_maj_real_boolean')

            self.plot_class.plt.show()
            # self.plot_class.save_hist_with_pickel(name=f'from_train_model_scan=True_{self.time_stamp}.pickle')
            # self.plot_class.save_fig(name=f'from_train_model_scan=True_{self.time_stamp}.png')

        else:
            self.plot_class.plot_each_hist((0, 0), name='train_loss')
            self.plot_class.plot_each_hist((0, 0), name='test_loss')
            self.plot_class.plot_each_hist((1, 0), name='train_acc')
            self.plot_class.plot_each_hist((1, 0), name='test_acc')
            self.plot_class.plot_each_hist((2, 0), name='train_auc')
            self.plot_class.plot_each_hist((2, 0), name='test_auc')

            self.plot_class.plt.show()
            # self.plot_class.save_hist_with_pickel(name=f'from_train_model_{self.time_stamp}.pickle')
            # self.plot_class.save_fig(name=f'from_train_model_{self.time_stamp}.png')

        # print('=====train========')
        # report_train_file = f'train_{self.time_stamp}'
        # report_test_file = f'test_{self.time_stamp}'
        # report_performance(y_true_dict['train_mask'], y_pred_dict['train_mask'],
        #                    y_score_dict['train_mask'],
        #                    labels=np.unique(self.data.y), verbose=True,
        #                    plot=True,
        #                    save_path=f'C:\\Users\\Anak\\PycharmProjects\\AdaptiveGraphStructureEmbedding\\Output\\Report\\train_model\\',
        #                    file_name=report_train_file)
        # print('=====test======')
        # report_performance(y_true_dict['test_mask'], y_pred_dict['test_mask'],
        #                    y_score_dict['test_mask'],
        #                    labels=np.unique(self.data.y), verbose=True,
        #                    plot=True,
        #                    save_path=f'C:\\Users\\Anak\\PycharmProjects\\AdaptiveGraphStructureEmbedding\\Output\\Report\\train_model\\',
        #                    file_name=report_test_file)

    def init_my_new_model(self):
        if self.log:
            log.info("in init_my_new_model")
        # =====================
        # ==hyper parameters setup
        # =====================
        self.num_batches = 1
        self.num_epochs = 200

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

    np.random.seed(111)
    data = preparing_cora_for_new_purposed_model(args.log)

    # todo here>> convert torch geometric data to torch data
    my_new_model = MyNewModel(data)
    my_new_model.run_my_new_model()
