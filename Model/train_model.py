import inspect
import itertools
import os
import sys
import time
import pandas as pd

from sklearn.model_selection import StratifiedKFold

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
# from Plot import PlotClass
from src.Visualization import PlotClass
# from src.Visualization.PlotEmb import plot_emb
from src.Visualization import PlotEmb
from src.Evaluation import get_total_roc_auc_score
from src.Preprocessing import ModelInputData
from src.Evaluation import report_performance


# warnings.simplefilter("error")
# warnings.simplefilter("ignore", DeprecationWarning)
# warnings.simplefilter("ignore", UserWarning)




class MyNewModel:
    # def __init__(self, data, dataset, dataloader):
    def __init__(self, dataset, main_epoch, num_gan_epoch, k_fold_split=3,
                 isLog=False, run_gcn_only=False, model_name=None,
                 device='cpu', downsample=True,
                 preserved_edges_percent=None):
        assert preserved_edges_percent is not None, "preserved_edges_percent must be specified to avoid ambiguity"
        assert model_name is not None, "model_name must be specified to avoid ambiguity"
        assert isinstance(dataset, str), ' please specify dataset '

        self.cur_dir = os.getcwd()
        self.preserved_edges_percent = preserved_edges_percent
        self.plot_emb = PlotEmb(args.save_emb)
        self.model_name = model_name
        self.num_gan_epoch = num_gan_epoch
        self.main_epoch = main_epoch
        self.is_downsampled = '' if not downsample else 'downsample'
        self.dataset = dataset

        self.device = device
        self.k_fold_split = k_fold_split
        self.time_stamp = time.strftime("%Y%m%d-%H%M%S")

        self.plot_class = PlotClass(save_status=args.save_plot)

        self.log = isLog
        self.dataset = dataset
        self.model_input_data = ModelInputData(self.cur_dir, self.dataset,
                                               downsample=downsample)
        self.data = self.model_input_data.data
        self.init_my_new_model()
        self.run_gcn_only = run_gcn_only
        self.total_accs_dict = {}
        self.total_aucs_dict = {}
        self.total_loss = {}

        # self.save_path = f'C:\\Users\\Anak\\PycharmProjects\\AdaptiveGraphStructureEmbedding\\Output\\Report\\{self.dataset}\\{self.model_name}\\'
        self.save_path = f'{self.cur_dir}\\..\\Output\\Report\\{self.dataset}\\{self.model_name}\\'


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
        self.gan = gan_model.GAN(self.data, device=self.device)
        self.gan.init_gan()

        # =====================
        # ==for Gcn
        # ====================
        self.gcn = gcn_model.GCN(self.data,
                                 preserved_percent=self.preserved_edges_percent,
                                 device=self.device)


    def run_my_new_model(self, ):

        skf = StratifiedKFold(n_splits=self.k_fold_split)

        self.gcn.randomedge_sampler()
        original_y = self.data.y
        for train, test in skf.split(np.arange(self.data.x.shape[0]),
                                     original_y):

            self.model_input_data.trainning_selected_ind = train
            self.model_input_data.test_selected_ind = test

            self.loss_per_epoch = {}
            self.accs_hist_dict = {}
            self.aucs_hist_dict = {}

            for epoch in range(self.main_epoch):
                name_and_val_dict = self.run_my_new_model_once(
                    epoch)

            #=====================
            #==avg
            #=====================

            self.get_total_loss()
            self.get_total_accs_and_aucs()

            #=====================
            #==plot  & report performance
            #=====================

            # self.plot_scan_and_loss(name_and_val_dict=None,
            #                         plot_scan=False,
            #                         plot_train_test=False,
            #                         plot_roc=False,
            #                         save_file=args.save_file,
            #                         save_plot=args.save_plot,
            #                         display_report=True)

        # =====================
        # ==plotting and report performance for cross validation
        # =====================

        self.collect_data_for_plotting()
        test_report, test_cm = self.plot_scan_and_loss(name_and_val_dict=name_and_val_dict,
                                plot_scan=True,
                                plot_train_test=True,
                                save_file=args.save_cv_file,
                                save_plot=args.save_cv_plot,
                                display_report=True,
                                return_report_stat_for_cv=True)
        avg_auc, avg_acc = test_report.loc['acc/total']['AUC'], test_report.loc['acc/total']['ACC']
        return avg_auc, avg_acc

    def run_my_new_model_once(self, epoch):
        if self.log:
            log.info('in run_my_new_model_once..')

        self.prep_train_test_data_index(epoch)

        name_and_val_dict = {}

        if not self.run_gcn_only:
            name_and_val_dict = self.run_gcn_gan_once( epoch, name_and_val_dict)
        else:
            name_and_val_dict = self.run_gcn_once( epoch, name_and_val_dict)

        self.loss_per_epoch.setdefault('train_loss', []).append(
            name_and_val_dict['train_loss'].cpu().detach().numpy())
        self.loss_per_epoch.setdefault('test_loss', []).append(
            name_and_val_dict['test_loss'].cpu().detach().numpy())


        return name_and_val_dict


    def run_gcn_once(self, epoch, name_and_val_dict):
        logits, trainning_loss, test_loss, emb = self.train_gcn_once(name_and_val_dict)

        # name_and_val_dict['y_true']  = torch.tensor(self.data.y).type(torch.long)
        name_and_val_dict['y_true']  = self.data.y
        name_and_val_dict['logits'] = logits
        self.get_result_per_epoch(name_and_val_dict)

        #=====================
        #==result and plot
        #=====================

        name_and_val_dict = self.collect_var_in_name_and_val_dict(name_and_val_dict, **{'train_loss':trainning_loss,
                                                                                        'test_loss':test_loss,
                                                                                        'emb': emb}
                                                                  )

        self.display_and_plot_result(epoch, name_and_val_dict)

        return name_and_val_dict

    def run_gcn_gan_once(self, epoch, name_and_val_dict):
        y_true, logits, trainning_loss, test_loss, emb = self.train_gcn_gan_once(epoch, name_and_val_dict)

        return name_and_val_dict

    def run_gan_components_of_new_model(self, num_gan_epoch):
        if self.log:
            log.info('in run_gan_components_of_new_model...')

        for i in range(num_gan_epoch):
            print(f'running gan {i}')
            for n_batch, (real_batch, y) in enumerate(
                    self.min_class_data_loader_for_gan):
                self.number_of_sample_per_batch = real_batch.size(0)
                real_data = real_batch

                fake_data = self.gan.generator(
                    gan_model.noise(self.number_of_sample_per_batch).to(
                        self.device))  # 10, 1433

                d_error, d_pred_real, d_pred_fake = \
                    self.gan.train_discriminator(self.gan.d_optimizer,
                                                 real_data,
                                                 fake_data)

                fake_data = self.gan.generator(gan_model.noise(
                    self.number_of_sample_per_batch).to(self.device))

                g_error = self.gan.train_generator(self.gan.g_optimizer,
                                                   fake_data.to(self.device))
                if self.log:
                    log.info(f'running GCN epoch = {n_batch}')


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
        self.min_class_data_loader_for_gan = torch.utils.data.DataLoader(
            self.min_class_cora_torch_dataset, batch_size=self.batch_size,
            shuffle=True)
        # return torch.utils.data.DataLoader(
        #     self.min_class_cora_torch_dataset, batch_size=self.batch_size,
        #     shuffle=True)


    def collect_performance(self, logits, y_true):

        y_pred_dict = {}
        y_score_dict = {}
        y_true_dict = {}

        accs_dict = {}
        aucs_dict = {}

        name_and_mask_dict = {
            'trainning_select_minreal_majreal_ind': self.data.trainning_select_minreal_majreal_ind,
            'test_select_minfake_minreal_majreal_ind': self.data.test_select_minfake_minreal_majreal_ind,
            'trainning_select_min_real_ind': self.data.trainning_select_min_real_ind,
            'trainning_select_maj_real_ind': self.data.trainning_select_maj_real_ind,
            'test_select_min_real_ind': self.data.test_select_min_real_ind,
            'test_select_maj_real_ind': self.data.test_select_maj_real_ind
        }
        print('hihi')

        # for name, mask in self.data('trainning_select_minreal_majreal_ind',
        #                             'test_select_minfake_minreal_majreal_ind',
        #                             'trainning_select_min_real_ind',
        #                             'trainning_select_maj_real_ind',
        #                             'test_select_min_real_ind',
        #                             'test_select_maj_real_ind',
        #                             ):
        for name, mask in name_and_mask_dict.items():

            pred = logits[mask].max(1)[1]
            y_true_mask = y_true[mask].cpu().detach().numpy()
            y_score = logits[mask].cpu().detach().numpy()

            y_pred_dict.setdefault(name, pred.cpu().detach().numpy())
            y_score_dict.setdefault(name, y_score)
            y_true_dict.setdefault(name, y_true_mask)

            acc = pred.eq(y_true_mask).sum().item() / mask.shape[0]
            accs_dict.setdefault(name, acc)

            if name in ['trainning_select_minreal_majreal_ind',
                        'test_select_minreal_majreal_ind']:
                auc = get_total_roc_auc_score(y_true_mask, y_score)
                aucs_dict.setdefault(name, auc)

        return accs_dict, aucs_dict, y_pred_dict, y_score_dict, y_true_dict

    def collect_data_for_plotting(self):
        def avg(x):
            return x / self.k_fold_split

        if not self.run_gcn_only:

            # TODO how to avg over list => sum element wise then divide by k_fold_split?
            avg_trainning_loss = avg(self.total_loss['trainning_loss'])
            avg_test_loss = avg(self.total_loss['test_loss'])
            avg_trainning_minfake_acc = avg(
                self.total_accs_dict['select_min_fake_ind'])
            avg_trainning_minreal_acc = avg(
                self.total_accs_dict['trainning_select_min_real_ind'])
            avg_trainning_majreal_acc = avg(
                self.total_accs_dict['trainning_select_maj_real_ind'])
            avg_trainning_acc = avg(self.total_accs_dict[
                                        'trainning_select_minfake_minreal_majreal_ind'])
            avg_trainning_auc = avg(self.total_aucs_dict[
                                        'trainning_select_minfake_minreal_majreal_ind'])

            avg_test_minreal_acc = avg(
                self.total_accs_dict['test_select_min_real_ind'])
            avg_test_majreal_acc = avg(
                self.total_accs_dict['test_select_maj_real_ind'])
            avg_test_acc = avg(
                self.total_accs_dict['test_select_minreal_majreal_ind'])
            avg_test_auc = avg(
                self.total_aucs_dict['test_select_minreal_majreal_ind'])

            print(f"""
                Average over {self.k_fold_split}
                Trainning: (loss: {avg_trainning_loss[-1]:.4f}, min_real: acc={avg_trainning_minreal_acc[-1]:.4f} maj: acc={avg_trainning_majreal_acc[-1]:.4f}  min_fake: acc={avg_trainning_minfake_acc[-1]:.4f})
                Test     : (loss: {avg_test_loss[-1]:.4f} min_real = acc={avg_test_minreal_acc[-1]:.4f} maj: acc={avg_test_majreal_acc[-1]:.4f}) 

                Train: (loss: {avg_trainning_loss[-1]:.4f}, acc={avg_trainning_acc[-1]:.4f}, auc={avg_trainning_auc[-1]: .4f})
                Test:  (loss: {avg_test_loss[-1]:.4f}, acc ={avg_test_acc[-1]:.4f}, auc={avg_test_auc[-1]:4f})
            """)

            self.plot_class.hist = {}
            dict_for_plot = {'train_loss': avg_trainning_loss,
                                 'train_acc': avg_trainning_acc,
                                 'trainning_select_min_real_ind': avg_trainning_minreal_acc,
                                 'trainning_select_maj_real_ind': avg_trainning_majreal_acc,
                                 'train_auc': avg_trainning_auc,
                                 'test_loss': avg_test_loss,
                                 'test_acc': avg_test_acc,
                                 'test_auc': avg_test_auc,
                                 'test_select_min_real_ind': avg_test_minreal_acc,
                                 'test_select_maj_real_ind': avg_test_majreal_acc,
                                 'select_min_fake_ind': avg_trainning_minfake_acc

                                 }
            self.plot_class.collect_hist_using_list_of_name(
                dict_for_plot=dict_for_plot)

        else:

            avg_trainning_loss = avg(self.total_loss['trainning_loss'])
            avg_test_loss = avg(self.total_loss['test_loss'])
            avg_trainning_minreal_acc = avg(
                self.total_accs_dict['trainning_select_min_real_ind'])
            avg_trainning_majreal_acc = avg(
                self.total_accs_dict['trainning_select_maj_real_ind'])
            avg_trainning_acc = avg(
                self.total_accs_dict['trainning_select_minreal_majreal_ind'])
            avg_trainning_auc = avg(
                self.total_aucs_dict['trainning_select_minreal_majreal_ind'])

            avg_test_minreal_acc = avg(
                self.total_accs_dict['test_select_min_real_ind'])
            avg_test_majreal_acc = avg(
                self.total_accs_dict['test_select_maj_real_ind'])
            avg_test_acc = avg(
                self.total_accs_dict['test_select_minreal_majreal_ind'])
            avg_test_auc = avg(
                self.total_aucs_dict['test_select_minreal_majreal_ind'])

            print(f"""
                Average over {self.k_fold_split}
                Trainning: (loss: {avg_trainning_loss[-1]:.4f}, min_real: acc={avg_trainning_minreal_acc[-1]:.4f} maj: acc={avg_trainning_majreal_acc[-1]:.4f})
                Test     : (loss: {avg_test_loss[-1]:.4f} min_real = acc={avg_test_minreal_acc[-1]:.4f} maj: acc={avg_test_majreal_acc[-1]:.4f}) 

                Train: (loss: {avg_trainning_loss[-1]:.4f}, acc={avg_trainning_acc[-1]:.4f}, auc={avg_trainning_auc[-1]: .4f})
                Test:  (loss: {avg_test_loss[-1]:.4f}, acc ={avg_test_acc[-1]:.4f}, auc={avg_test_auc[-1]:4f})
            """)

            self.plot_class.hist = {}
            dict_for_plot = {'train_loss': avg_trainning_loss,
                                 'train_acc': avg_trainning_acc,
                                 'trainning_select_min_real_ind': avg_trainning_minreal_acc,
                                 'trainning_select_maj_real_ind': avg_trainning_majreal_acc,
                                 'train_auc': avg_trainning_auc,
                                 'test_loss': avg_test_loss,
                                 'test_acc': avg_test_acc,
                                 'test_auc': avg_test_auc,
                                 'test_select_min_real_ind': avg_test_minreal_acc,
                                 'test_select_maj_real_ind': avg_test_majreal_acc,
                                 }
            self.plot_class.collect_hist_using_list_of_name(
                dict_for_plot=dict_for_plot)

    def get_result_per_epoch(self, name_and_val_dict, is_train=None):

        if not self.run_gcn_only:
            assert isinstance(is_train, bool), ''
            y_true_dict, y_pred_dict, y_score_dict, y_score_dict, accs_dict, aucs_dict = self.get_result_of_gcn_gan(name_and_val_dict, is_train)
        else:
            y_true_dict, y_pred_dict, y_score_dict, y_score_dict, accs_dict, aucs_dict = self.get_result_of_gcn(name_and_val_dict)

        name_and_val_dict.setdefault('accs_dict', {}).update(accs_dict)
        name_and_val_dict.setdefault('aucs_dict', {}).update(aucs_dict)
        name_and_val_dict.setdefault('y_true_dict', {}).update(y_true_dict)
        name_and_val_dict.setdefault('y_pred_dict', {}).update(y_pred_dict)
        name_and_val_dict.setdefault('y_score_dict', {}).update(y_score_dict)

    def get_total_loss(self):
        if 'trainning_loss' not in self.total_loss:
            self.total_loss['trainning_loss'] = np.array(
                self.loss_per_epoch['train_loss'])
        else:
            self.total_loss['trainning_loss'] += self.loss_per_epoch[
                'test_loss']
        if 'test_loss' not in self.total_loss:
            self.total_loss['test_loss'] = np.array(
                self.loss_per_epoch['test_loss'])
        else:
            self.total_loss['test_loss'] += self.loss_per_epoch[
                'test_loss']

    def get_total_accs_and_aucs(self):
        for i, j in self.accs_hist_dict.items():
            x = np.array(j)
            if i not in self.total_accs_dict:
                self.total_accs_dict[i] = x
            else:
                self.total_accs_dict[i] += x

        for i, j in self.aucs_hist_dict.items():
            x = np.array(j)
            if i not in self.total_aucs_dict:
                self.total_aucs_dict[i] = x
            else:
                self.total_aucs_dict[i] += x

    def train_gcn_once(self, name_and_val_dict):

        self.gcn.model.train()

        train_emb_after_conv1, train_emb_after_conv2, logits = self.gcn.model(
            self.gcn.get_dgl_graph(),
            self.data.x,
            run_all=True)
        trainning_loss = self.gcn.loss_and_step(
            logits[self.model_input_data.trainning_selected_ind],
            self.data.y[
                self.model_input_data.trainning_selected_ind])

        # =====================
        # == gcn test
        # =====================
        self.gcn.model.eval()

        test_emb_after_conv1, test_emb_after_conv2, logits = self.gcn.model(
            self.gcn.get_dgl_graph(), self.data.x,
            run_all=True)

        test_loss = self.gcn.loss_and_step(
            logits[self.model_input_data.test_selected_ind],
            self.data.y[
                self.model_input_data.test_selected_ind])

        self.collect_var_in_name_and_val_dict(name_and_val_dict, **{'trainning_loss': trainning_loss,
                                                                    'test_loss': test_loss})


        return logits, trainning_loss, test_loss, train_emb_after_conv2

    def train_gcn_gan_once(self, epoch, name_and_val_dict):
        # self.apply_trianning()
        self.gcn.model.train()

        train_emb_after_conv1 = self.gcn.model(self.gcn.get_dgl_graph(),
                                               self.data.x.to(self.device),
                                               get_conv1_emb=True)

        # =====================
        # ==gan dataset
        # =====================
        self.prepare_gan_trainning_dataset(
            train_emb_after_conv1,
            self.model_input_data.trainning_selected_min_ind)

        self.run_gan_components_of_new_model(
            num_gan_epoch=self.num_gan_epoch)

        # =====================
        # == fake_neg/true_neg/pos = 25%/25%/50%
        # =====================
        fake_data = self.gan.generator(gan_model.noise(
            self.model_input_data.trainning_selected_min_ind.shape[
                0]).to(
            self.device))  # this will be sent to discriminator 2 too

        minreal_minfake_majreal_x = torch.cat(
            (train_emb_after_conv1, fake_data), 0).to(self.device)
        minreal_minfake_majreal_y = torch.cat(
            (self.data.y,
             torch.zeros(
                 fake_data.size(0)).type(
                 torch.long)), 0).type(
            torch.long).to(self.device)

        if epoch == 0:
            self.model_input_data.set_data(fake_data=fake_data)
            self.model_input_data.set_train_test_data_index()
        self.prepare_data_ind()

        train_emb_after_conv2, train_emb_after_conv2_with_external_input, logits = self.gcn.model(
            self.gcn.get_dgl_graph(),
            train_emb_after_conv1,
            external_input=fake_data.to(
                self.device),
            run_discriminator=True)
        trainning_loss = self.gcn.loss_and_step(
            logits[self.data.trainning_select_minfake_minreal_majreal_ind],
            minreal_minfake_majreal_y[
                self.data.trainning_select_minfake_minreal_majreal_ind])

        y_true = minreal_minfake_majreal_y

        name_and_val_dict['logits'] = logits
        name_and_val_dict['y_true'] = y_true
        self.get_result_per_epoch(name_and_val_dict, is_train=True)


        # =====================
        # == gcn test
        # =====================
        self.gcn.model.eval()

        test_emb_after_conv1, test_emb_after_conv2, logits = self.gcn.model(
            self.gcn.get_dgl_graph(), self.data.x.to(self.device),
            run_all=True)

        test_loss = self.gcn.loss(
            logits[self.data.test_selected_ind],
            self.data.y[
                self.data.test_selected_ind].to(self.device))



        name_and_val_dict['y_true']  = y_true
        name_and_val_dict['logits'] = logits
        self.get_result_per_epoch(name_and_val_dict, is_train=False)

        #=====================
        #==Result and performance
        #=====================
        name_and_val_dict = self.collect_var_in_name_and_val_dict(name_and_val_dict,**{'train_loss': trainning_loss,
                                                                                       'test_loss': test_loss,
                                                                                       'emb': train_emb_after_conv2_with_external_input}
                                                                  )

        self.display_and_plot_result(epoch, name_and_val_dict)
        return y_true, logits, trainning_loss, test_loss, train_emb_after_conv2

    def plot_scan_and_loss(self, name_and_val_dict=None,
                           plot_scan=False,
                           plot_train_test=False,
                           plot_roc=False,
                           save_file=False,
                           save_plot=False,
                           display_report=False,
                           return_report_stat_for_cv=False):

        """when save_plot is True, it only svae plot that its status is True"""

        self.plot_class.save_status = save_plot

        if args.run_gcn_only:
            model_name = 'run_gcn'
        elif args.run_gcn_gan:
            model_name = 'train_model'

        naming_convention_dict = {
            'dataset': self.dataset,
            'model_name': self.model_name,
            'is_downsampled': self.is_downsampled,
            'main_epoch': self.main_epoch,
            'num_gan_epoch': self.num_gan_epoch,
            'time_stamp': self.time_stamp,
            'preserved_edges_percent': self.preserved_edges_percent
        }

        #=====================
        #==plot emb
        #=====================
        save_path = self.save_path
        title = create_file_naming_convension(naming_convention_dict, title=True)
        emb_file_name = create_file_naming_convension(naming_convention_dict, emb=True)
        scan_file_name = create_file_naming_convension(naming_convention_dict,
                                                       scan=True)
        train_test_name = create_file_naming_convension(naming_convention_dict,
                                                        train_test=True)
        report_train_file = create_file_naming_convension(
            naming_convention_dict,
            report=True, is_train=True)
        report_test_file = create_file_naming_convension(naming_convention_dict,
                                                         report=True,
                                                         is_train=False)
        print(save_path)
        print(title)
        print(emb_file_name)
        print(scan_file_name)
        print(train_test_name)
        print(report_train_file)
        print(report_test_file)

        if name_and_val_dict['emb'] is not None:
            emb_dict = self.apply_tsne_on_emb(
                name_and_val_dict['emb'].cpu().detach().numpy(), run_gcn_only=self.run_gcn_only)
            for i, j in emb_dict.items():
                self.plot_emb.collect_emb(i, j)
            self.plot_emb.plot_all(save_path=save_path, title=emb_file_name)

        #=====================
        #==plot other thing + report performance
        #=====================


        if not self.run_gcn_only:

            if plot_scan:
                list_of_name = {'train_loss': (0, 0),
                                'test_loss': (0, 0),
                                'trainning_select_min_real_ind': (1, 0),
                                'select_min_fake_ind': [(1, 0), (2, 0)],
                                'trainning_select_maj_real_ind': (1, 0),
                                'test_select_min_real_ind': (2, 0),
                                'test_select_maj_real_ind': (2, 0),
                                }
                self.plot_class.plot_using_list_of_name(subplot_size=(3, 1),
                                                        name_and_tuple_dict=list_of_name,
                                                        save_file_name=scan_file_name,
                                                        title= title,
                                                        save_path=save_path)

            if plot_train_test:

                list_of_name = {'train_loss': (0, 0),
                                'train_acc': (1, 0),
                                'train_auc': (2, 0),
                                'test_loss': (0, 0),
                                'test_acc': (1, 0),
                                'test_auc': (2, 0),
                                }

                self.plot_class.plot_using_list_of_name(subplot_size=(3, 1),
                                                        name_and_tuple_dict=list_of_name,
                                                        save_file_name=train_test_name,
                                                        title=title,
                                                        save_path=save_path)

            print('=====train========')
            train_performance = report_performance(
                name_and_val_dict['y_true_dict'][
                    'trainning_select_minfake_minreal_majreal_ind'],
                name_and_val_dict['y_pred_dict'][
                    'trainning_select_minfake_minreal_majreal_ind'],
                name_and_val_dict['y_score_dict'][
                    'trainning_select_minfake_minreal_majreal_ind'],
                labels=np.unique(self.data.y), verbose=display_report,
                plot=plot_roc,
                save_status=save_file,
                save_path= save_path,
                file_name=report_train_file,
                return_value_for_cv=return_report_stat_for_cv
            )
            train_report, train_cm = (
                None, None) if train_performance is None else train_performance
            print('=====test======')
            test_performance = report_performance(
                name_and_val_dict['y_true_dict']['test_select_minreal_majreal_ind'],
                name_and_val_dict['y_pred_dict']['test_select_minreal_majreal_ind'],
                name_and_val_dict['y_score_dict']['test_select_minreal_majreal_ind'],
                labels=np.unique(self.data.y), verbose=display_report,
                plot=plot_roc,
                save_status=save_file,
                save_path=save_path,
                file_name=report_test_file,
                return_value_for_cv=return_report_stat_for_cv)

            test_report, test_cm = (
                None, None) if test_performance is None else test_performance
        else:
            if plot_scan:
                list_of_name = {'train_loss': (0, 0),
                                'test_loss': (0, 0),
                                'trainning_select_min_real_ind': (1, 0),
                                'trainning_select_maj_real_ind': (1, 0),
                                'test_select_min_real_ind': (2, 0),
                                'test_select_maj_real_ind': (2, 0),
                                }

                self.plot_class.plot_using_list_of_name(subplot_size=(3, 1),
                                                        name_and_tuple_dict=list_of_name,
                                                        save_file_name=scan_file_name,
                                                        title=title,
                                                        save_path=save_path)

            if plot_train_test:
                list_of_name = {'train_loss': (0, 0),
                                'train_acc': (1, 0),
                                'train_auc': (2, 0),
                                'test_loss': (0, 0),
                                'test_acc': (1, 0),
                                'test_auc': (2, 0),
                                }

                self.plot_class.plot_using_list_of_name(subplot_size=(3, 1),
                                                        name_and_tuple_dict=list_of_name,
                                                        save_file_name=train_test_name,
                                                        title=title,
                                                        save_path=save_path)

            print('=====train========')
            train_performance = report_performance(
                name_and_val_dict['y_true_dict']['trainning_select_minreal_majreal_ind'],
                name_and_val_dict['y_pred_dict']['trainning_select_minreal_majreal_ind'],
                name_and_val_dict['y_score_dict']['trainning_select_minreal_majreal_ind'],
                labels=np.unique(self.data.y), verbose=display_report,
                plot=plot_roc,
                save_status=save_file,
                save_path=save_path,
                file_name=report_train_file,
                return_value_for_cv=return_report_stat_for_cv)
            train_report, train_cm = (
                None, None) if train_performance is None else train_performance
            print('=====test======')
            test_performance = report_performance(
                name_and_val_dict['y_true_dict']['test_select_minreal_majreal_ind'],
                name_and_val_dict['y_pred_dict']['test_select_minreal_majreal_ind'],
                name_and_val_dict['y_score_dict']['test_select_minreal_majreal_ind'],
                labels=np.unique(self.data.y), verbose=display_report,
                plot=plot_roc,
                save_status=save_file,
                save_path=save_path,
                file_name=report_test_file,
                return_value_for_cv=return_report_stat_for_cv)
            test_report, test_cm = (
                None, None) if test_performance is None else test_performance
        return test_report, test_cm

    def display_and_plot_result(self, epoch, name_and_val_dict):

        self.display_result_per_epoch(epoch, name_and_val_dict)

    def display_result_per_epoch(self, epoch, name_and_val_dict, ):
        if not self.run_gcn_only:

            print(f"""
                Epoch: {epoch:03d},
                Trainning: (loss: {name_and_val_dict['train_loss']:.4f}, min_real: acc={name_and_val_dict['accs_dict']["trainning_select_min_real_ind"]:.4f} maj: acc={name_and_val_dict['accs_dict']["trainning_select_maj_real_ind"]:.4f})  min_fake: acc={name_and_val_dict['accs_dict']["select_min_fake_ind"]:.4f}
                Test     : (loss: {name_and_val_dict['test_loss']:.4f} min_real = acc={name_and_val_dict['accs_dict']["test_select_min_real_ind"]:.4f} maj: acc={name_and_val_dict['accs_dict']["test_select_maj_real_ind"]:.4f}) 

                Train: (acc={name_and_val_dict['accs_dict']["trainning_select_minfake_minreal_majreal_ind"]:.4f}, auc={name_and_val_dict['aucs_dict']["trainning_select_minfake_minreal_majreal_ind"]: .4f}, loss={name_and_val_dict['train_loss']:.4f})
                Test: ({name_and_val_dict['accs_dict']["test_select_minreal_majreal_ind"]:.4f}, auc={name_and_val_dict['aucs_dict']["test_select_minreal_majreal_ind"]:4f}, loss={name_and_val_dict['test_loss']:.4f})
            """)
        else:
            print(f"""
                Epoch: {epoch:03d},
                Trainning: (loss: {name_and_val_dict['train_loss']:.4f}, min_real: acc={name_and_val_dict['accs_dict']["trainning_select_min_real_ind"]:.4f} maj: acc={name_and_val_dict['accs_dict']["trainning_select_maj_real_ind"]:.4f})
                Test     : (loss: {name_and_val_dict['test_loss']:.4f} min_real: acc={name_and_val_dict['accs_dict']["test_select_min_real_ind"]:.4f} maj: acc={name_and_val_dict['accs_dict']["test_select_maj_real_ind"]:.4f}) 
                Train: (acc={name_and_val_dict['accs_dict']["trainning_select_minreal_majreal_ind"]:.4f}, auc={name_and_val_dict['aucs_dict']["trainning_select_minreal_majreal_ind"]: .4f}, loss={name_and_val_dict['train_loss']:.4f})
                Test: ({name_and_val_dict['accs_dict']["test_select_minreal_majreal_ind"]:.4f}, auc={name_and_val_dict['aucs_dict']["test_select_minreal_majreal_ind"]:4f}, loss={name_and_val_dict['test_loss']:.4f})
            """)

    def prep_train_test_data_index(self,epoch):

        if epoch == 0:
            self.model_input_data.set_data()
            self.model_input_data.set_train_test_data_index()

        self.data.trainning_select_min_real_ind = self.model_input_data.trainning_select_min_real_ind
        self.data.trainning_select_maj_real_ind = self.model_input_data.trainning_select_maj_real_ind
        self.data.trainning_selected_ind = self.model_input_data.trainning_selected_ind

        self.data.test_select_min_real_ind = self.model_input_data.test_select_min_real_ind
        self.data.test_select_maj_real_ind = self.model_input_data.test_select_maj_real_ind
        self.data.test_selected_ind = self.model_input_data.test_selected_ind

        self.data.trainning_select_minreal_majreal_ind = self.model_input_data.trainning_select_minreal_majreal_ind
        self.data.test_select_minreal_majreal_ind = self.model_input_data.test_select_minreal_majreal_ind

    def prepare_data_ind(self):

        self.data.trainning_select_min_real_ind = self.model_input_data.trainning_select_min_real_ind
        self.data.trainning_select_maj_real_ind = self.model_input_data.trainning_select_maj_real_ind
        self.data.trainning_selected_ind = self.model_input_data.trainning_selected_ind

        self.data.test_select_min_real_ind = self.model_input_data.test_select_min_real_ind
        self.data.test_select_maj_real_ind = self.model_input_data.test_select_maj_real_ind
        self.data.test_selected_ind = self.model_input_data.test_selected_ind

        self.data.select_min_fake_ind = self.model_input_data.select_min_fake_ind

        self.data.trainning_select_minreal_majreal_ind = self.model_input_data.trainning_select_minreal_majreal_ind
        self.data.trainning_select_minfake_minreal_majreal_ind = self.model_input_data.trainning_select_minfake_minreal_majreal_ind
        self.data.test_select_minreal_majreal_ind = self.model_input_data.test_select_minreal_majreal_ind
        self.data.test_select_minfake_minreal_majreal_ind = self.model_input_data.test_select_minfake_minreal_majreal_ind

    def get_result_of_gcn_gan(self, name_and_val_dict, is_train):

        y_pred_dict = {}
        y_score_dict = {}
        y_true_dict = {}
        accs_dict = {}
        aucs_dict = {}

        if is_train :
            name_and_mask_dict = {
                'trainning_select_minfake_minreal_majreal_ind': self.data.trainning_select_minfake_minreal_majreal_ind,
                'trainning_select_minreal_majreal_ind': self.data.trainning_select_minreal_majreal_ind,
                'trainning_select_min_real_ind': self.data.trainning_select_min_real_ind,
                'trainning_select_maj_real_ind': self.data.trainning_select_maj_real_ind,
                'select_min_fake_ind': self.data.select_min_fake_ind
                # 'test_select_minfake_minreal_majreal_ind': self.data.test_select_minfake_minreal_majreal_ind,
                # 'test_select_min_real_ind': self.data.test_select_min_real_ind,
                # 'test_select_maj_real_ind': self.data.test_select_maj_real_ind
            }
            for name, mask in name_and_mask_dict.items():

                pred = name_and_val_dict['logits'][mask].max(1)[1]
                y_true_mask = name_and_val_dict['y_true'][mask]
                y_score = name_and_val_dict['logits'][mask]

                y_pred_dict.setdefault(name, pred)
                y_score_dict.setdefault(name, y_score)
                y_true_dict.setdefault(name, y_true_mask)

                acc = pred.eq(y_true_mask).sum().item() / mask.shape[0]
                accs_dict.setdefault(name, acc)
                self.accs_hist_dict.setdefault(name, []).append(acc)

                if name in ['trainning_select_minfake_minreal_majreal_ind',
                            'test_select_minreal_majreal_ind']:
                    auc = get_total_roc_auc_score(
                        y_true_mask.cpu().detach().numpy(), y_score.cpu().detach().numpy())
                    auc = torch.tensor(auc).type(torch.long)
                    aucs_dict.setdefault(name, auc)
                    self.aucs_hist_dict.setdefault(name, []).append(auc)

        else:
            name_and_mask_dict = {
                # 'test_select_minfake_minreal_majreal_ind': self.data.test_select_minfake_minreal_majreal_ind,
                'test_select_minreal_majreal_ind': self.data.test_select_minreal_majreal_ind,
                'test_select_min_real_ind': self.data.test_select_min_real_ind,
                'test_select_maj_real_ind': self.data.test_select_maj_real_ind,
                # 'select_min_fake_ind': self.data.select_min_fake_ind
            }

            for name, mask in name_and_mask_dict.items():

                pred = name_and_val_dict['logits'][mask].max(1)[1]
                y_true_mask = name_and_val_dict['y_true'][mask]
                y_score = name_and_val_dict['logits'][
                    mask]

                y_pred_dict.setdefault(name, pred)
                y_score_dict.setdefault(name, y_score)
                y_true_dict.setdefault(name, y_true_mask)

                acc = pred.eq(y_true_mask).sum().item() / mask.shape[0]
                accs_dict.setdefault(name, acc)
                self.accs_hist_dict.setdefault(name, []).append(acc)

                if name in ['trainning_select_minreal_majreal_ind',
                            'test_select_minreal_majreal_ind']:
                    auc = get_total_roc_auc_score(
                        y_true_mask.cpu().detach().numpy(),
                        y_score.cpu().detach().numpy())
                    auc = torch.tensor(auc).type(torch.long)
                    aucs_dict.setdefault(name, auc)
                    self.aucs_hist_dict.setdefault(name, []).append(auc)

        return y_true_dict, y_pred_dict, y_score_dict, y_score_dict, accs_dict, aucs_dict

    def get_result_of_gcn(self, name_and_val_dict):

        y_pred_dict = {}
        y_score_dict = {}
        y_true_dict = {}
        accs_dict = {}
        aucs_dict = {}

        name_and_mask_dict = {
            'trainning_select_minreal_majreal_ind': self.data.trainning_select_minreal_majreal_ind,
            'trainning_select_min_real_ind': self.data.trainning_select_min_real_ind,
            'trainning_select_maj_real_ind': self.data.trainning_select_maj_real_ind,
            'test_select_minreal_majreal_ind': self.data.test_select_minreal_majreal_ind,
            'test_select_min_real_ind': self.data.test_select_min_real_ind,
            'test_select_maj_real_ind': self.data.test_select_maj_real_ind
        }

        for name, mask in name_and_mask_dict.items():

            pred = name_and_val_dict['logits'][mask].max(1)[1]
            y_true_mask = name_and_val_dict['y_true'][mask]
            y_score = name_and_val_dict['logits'][mask]

            y_pred_dict.setdefault(name, pred)
            y_score_dict.setdefault(name, y_score)
            y_true_dict.setdefault(name, y_true_mask)

            acc = pred.eq(y_true_mask).sum().item() / mask.shape[0]
            accs_dict.setdefault(name, acc)
            self.accs_hist_dict.setdefault(name, []).append(acc)

            if name in ['trainning_select_minreal_majreal_ind',
                        'test_select_minreal_majreal_ind']:
                auc = get_total_roc_auc_score(y_true_mask.cpu().detach().numpy(), y_score.cpu().detach().numpy())
                auc = torch.tensor(auc).type(torch.long)
                aucs_dict.setdefault(name, auc)
                self.aucs_hist_dict.setdefault(name, []).append(auc)

        return y_true_dict, y_pred_dict, y_score_dict, y_score_dict, accs_dict, aucs_dict

    def apply_tsne_on_emb(self, emb, run_gcn_only=None):
        assert run_gcn_only is not None, "run_gcn_only must be specified to avoid ambiguity"
        emb_dict = {}

        from sklearn.manifold import TSNE

        emb_2d = TSNE(n_components=2).fit_transform(emb)
        print(emb_2d.shape)
        # output (4, 2)
        if not run_gcn_only:
            emb_dict.setdefault('test', emb_2d[self.data.test_selected_ind])
            emb_dict.setdefault('train', emb_2d[self.data.trainning_selected_ind])
            emb_dict.setdefault('min_real', np.concatenate((emb_2d[
                                                                self.data.trainning_select_min_real_ind.cpu().detach().numpy()],
                                                            emb_2d[
                                                                self.data.test_select_min_real_ind.cpu().detach().numpy()]),
                                                           axis=0))
            emb_dict.setdefault('min_fake', emb_2d[self.data.select_min_fake_ind])
            emb_dict.setdefault('maj', np.concatenate((emb_2d[
                                                           self.data.trainning_select_maj_real_ind.cpu().detach().numpy()],
                                                       emb_2d[
                                                           self.data.test_select_maj_real_ind.cpu().detach().numpy()]),
                                                      0))
        else:
            emb_dict.setdefault('test', emb_2d[self.data.test_selected_ind])
            emb_dict.setdefault('train', emb_2d[self.data.trainning_selected_ind])
            emb_dict.setdefault('min_real', np.concatenate((emb_2d[
                                                                self.data.trainning_select_min_real_ind.cpu().detach().numpy()],
                                                            emb_2d[
                                                                self.data.test_select_min_real_ind.cpu().detach().numpy()]),
                                                           axis=0))
            emb_dict.setdefault('maj', np.concatenate((emb_2d[
                                                           self.data.trainning_select_maj_real_ind.cpu().detach().numpy()],
                                                       emb_2d[
                                                           self.data.test_select_maj_real_ind.cpu().detach().numpy()]),
                                                      0))

        return emb_dict

    def collect_var_in_name_and_val_dict(self, dictionary, **kwargs):
        for key, val in kwargs.items():
            dictionary[key] = val
        return dictionary



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


def get_folder(condition_to_consider):
    assert isinstance(condition_to_consider, dict), ""
    import os


    cur_dir =  os.getcwd()
    folder = cur_dir + '/../Report/'
    if condition_to_consider['dataset'] == 'cora':
        dataset = 'cora/'
    elif condition_to_consider['dataset'] == 'citeseer':
        dataset = 'citeseer'
    else:
        raise ValueError("")

    if condition_to_consider['model_name'] == 'train_model':
        model_name = 'train_model'
    elif condition_to_consider['model_name'] == 'run_gcn':
        model_name = 'run_gcn'
    else:
        raise ValueError('')

    folder = folder + f'/{dataset}/{model_name}'
    return folder

def create_file_naming_convension(condition_to_consider, title=None, emb=None, is_train=None, scan=None, train_test=None, report=None):
    report_name = ''
    emb_name = ''
    scan_name = ''
    train_test_name = ''
    is_train_name = ''
    if title is not None:
        assert emb is None, ''
        assert scan is None, ''
        assert train_test is None, ''
        assert report is None, ''
    elif report is not None:
        assert is_train is not None,''
        is_train_name = 'train_' if is_train else 'test_'
        assert emb is None, ''
        assert scan is None, ''
        assert train_test is None, ''
    elif emb is not None:
        emb_name = 'emb_'
        assert report is None, ''
        assert scan is None, ''
        assert train_test is None, ''
    elif scan is not None:
        scan_name = 'scan_'
        assert emb is None, ''
        assert report is None, ''
        assert train_test is None, ''
    elif train_test is not None:
        train_test_name = 'train_test_'
        assert emb is None, ''
        assert scan is None, ''
        assert report is None, ''
    else:
        raise ValueError('')

    assert isinstance(condition_to_consider, dict), ""
    # assert time_stamp is not None, "time_stamp must be specified to avoid ambiguity"

    if condition_to_consider['model_name'] == 'train_model':
        model_name = 'train_model_'
    elif condition_to_consider['model_name'] == 'run_gcn':
        model_name = 'run_gcn_'
    else:
        raise ValueError('')

    if condition_to_consider['is_downsampled']:
        is_downsampled = 'downsampled_'
    else:
        is_downsampled = ''

    label = emb_name + report_name + scan_name + train_test_name + is_train_name
    file_name = f"{condition_to_consider['time_stamp']}_{model_name}_edge_percent={condition_to_consider['preserved_edges_percent']}_ep={condition_to_consider['main_epoch']}_gan_ep={condition_to_consider['num_gan_epoch']}_{is_downsampled}_{label}"
    return file_name

if __name__ == '__main__':
    # dataset = 'Cora'
    # path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'Data',
    #                 'External')
    #
    # dataset = Planetoid(path, dataset, T.NormalizeFeatures())
    # data = dataset[0]
    #
    np.random.seed(111)
    torch.manual_seed(111)

    # log = Logging.Logger(name=f'log_for_{model_name}_file')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if args.manual:
        """--manual -pasac"""
        main_epoch = [50, 100, 200]
        num_gan_epoch =  [1,5,10,25,50]
        preserved_edges_percent = [1]
        model_name_list = ['train_model', 'run_gcn']
        dataset_list = ['cora']

        # main_epoch = [1, 2]
        # num_gan_epoch =  [1]
        # preserved_edges_percent = [1]
        # model_name_list = ['train_model', 'run_gcn']
        # dataset_list = ['cora']

        epoch_pair = itertools.product(main_epoch, num_gan_epoch, preserved_edges_percent, model_name_list, dataset_list)

        model_performance_dict = {}
        for i,(e, ge, percent, model_name, dataset) in enumerate(epoch_pair):
            args.main_epoch = e
            args.num_gan_epoch = ge
            args.preserved_edges_percent = percent
            args.dataset = dataset
            my_new_model = MyNewModel(dataset, main_epoch=args.main_epoch,
                                      num_gan_epoch=args.num_gan_epoch,
                                      k_fold_split=args.k_fold_split,
                                      run_gcn_only=args.run_gcn_only,
                                      model_name=model_name,
                                      downsample=args.downsample,
                                      device=device,
                                      preserved_edges_percent=args.preserved_edges_percent)
            avg_auc, avg_acc = my_new_model.run_my_new_model()
            model_performance_dict[f'({model_name}_epoch={e}_gan_epoch={ge}_edge_percent={percent})'] = {'avg_auc':avg_auc, 'avg_acc':avg_acc}

        file_path = os.path.dirname(os.getcwd())+ f'\\Output\\Report\\{args.dataset}\\{my_new_model.time_stamp}_experiment.csv'
        print(f'save experiment to {file_path}')
        pd.DataFrame.from_dict(model_performance_dict).transpose().to_csv(file_path)
    else:
        """--run_gcn_gan -ng 1 -me 3 -pep 0.3 -pasac"""
        dataset = args.dataset  # cora, citeseer
        model_name = 'run_gcn' if args.run_gcn_only else 'train_model'
        my_new_model = MyNewModel(dataset, main_epoch=args.main_epoch,
                                  num_gan_epoch=args.num_gan_epoch,
                                  k_fold_split=args.k_fold_split,
                                  run_gcn_only=args.run_gcn_only,
                                  model_name=model_name,
                                  downsample=args.downsample,
                                  device=device,
                                  preserved_edges_percent=args.preserved_edges_percent)
        avg_auc, avg_acc = my_new_model.run_my_new_model()

