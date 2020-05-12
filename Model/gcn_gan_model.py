import pandas as pd
import os
import time

import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader
from arg_parser import args

import src.Modeling.gan as  gan_model
import src.Modeling.gcn as gcn_model
# from src.Evaluation import PerformanceTracker
from src.Evaluation import ReportPerformance
from src.Visualization import PlotPerformance
from src.Evaluation import get_total_roc_auc_score
from src.Modeling.my_new_model import MyNewModel
from src.Preparation.Data import CoraTorchDataset
from src.Preprocessing import ModelInputData
from src.Visualization import PlotEmb
from src.Preprocessing.prepare_emb import apply_tsne_on_emb


class GCNGAN(MyNewModel):
    def __init__(self, dataset_dict, model_parameters_dict, boolean_dict):
        super(GCNGAN, self).__init__(model_parameters_dict['run_gcn'])

        self.performance_per_epoch = {}
        self.check_input_requirement(dataset_dict, model_parameters_dict,
                                     boolean_dict)
        # =====================
        # ==unrolled dict
        # =====================
        self.dataset_dict = dataset_dict
        self.model_parameters_dict = model_parameters_dict
        self.boolean_dict = boolean_dict

        # =====================
        # ==class specific parameter
        # =====================
        self.cur_dir = os.getcwd()
        self.save_path = f'{self.cur_dir}\\..\\Output\\Report\\{self.dataset_dict["dataset"]}\\{self.model_parameters_dict["model_name"]}\\'

        # =====================
        # ==call Plotting class
        # =====================
        # self.plot_each_emb = PlotEmb(
        #     boolean_dict['plot_dict']['is_plotted_each_emb'],
        #     boolean_dict['save_dict']['is_saved_each_emb_plot'],
        # )
        self.plot_cv_emb = PlotEmb(
            boolean_dict['plot_dict']['is_plotted_cv_emb'],
            boolean_dict['save_dict']['is_saved_cv_emb_plot'],
        )

        # =====================
        # ==call plot performance
        # =====================
        self.plot_gan_performance = PlotPerformance(
            # PlotPerformance
            is_plotted=boolean_dict['plot_dict']['is_plotted_cv_performance'],
            is_saved_plot=boolean_dict['save_dict'][
                'is_saved_cv_performance_plot'])

        self.plot_cv_performance = PlotPerformance(
            # PlotPerformance
            is_plotted=boolean_dict['plot_dict']['is_plotted_cv_performance'],
            is_saved_plot=boolean_dict['save_dict'][
                'is_saved_cv_performance_plot'])

        self.report_cv_performance = ReportPerformance(
            # ReportPerformance
            is_displayed_performance_table=boolean_dict['plot_dict'][
                'is_displayed_cv_performance_table'],
            is_plotted_roc=boolean_dict['plot_dict']['is_plotted_cv_roc'],
            is_saved_performance_table=boolean_dict['save_dict'][
                'is_saved_cv_performance_table'],
            is_saved_plotted_roc=boolean_dict['save_dict'][
                'is_saved_cv_roc_plot']
        )

        # self.performance_tracker = PerformanceTracker(
        #     # PlotPerformance
        #     is_plotted=boolean_dict['plot_dict']['is_plotted_cv_performance'],
        #     is_saved_plot=boolean_dict['save_dict'][
        #         'is_saved_cv_performance_plot'],
        #     # ReportPerformance
        #     is_displayed_performance_table=boolean_dict['plot_dict'][
        #         'is_displayed_cv_performance_table'],
        #     is_plotted_roc=boolean_dict['plot_dict']['is_plotted_cv_roc'],
        #     is_saved_performance_table=boolean_dict['save_dict'][
        #         'is_saved_cv_performance_table'],
        #     is_saved_plotted_roc=boolean_dict['save_dict'][
        #         'is_saved_cv_roc_plot']
        # )

        # self.plot_each_performance = PlotPerformance(
        #     self.boolean_dict['plot_dict']['is_plotted_each_performance'],
        #     self.boolean_dict['save_dict']['is_saved_each_performance']
        # )
        # self.plot_cv_performance = PlotPerformance(
        #     self.boolean_dict['plot_dict']['is_plotted_cv_performance'],
        #     self.boolean_dict['save_dict']['is_saved_cv_performance']
        # )

        # =====================
        # ==get Preprocessed data given a dataset
        # =====================
        self.model_input_data = ModelInputData(self.cur_dir,
                                               self.dataset_dict['dataset'],
                                               is_downsampled=self.dataset_dict[
                                                   'is_downsampled'])
        # self.data = self.model_input_data.data

        self.init_model()
        self.total_accs_dict = {}
        self.total_aucs_dict = {}
        self.total_loss = {}
        #=====================
        #== multi index for experiemtns
        #=====================
        self.create_multi_level_row_index()
        self.create_multi_level_col_index()

    def check_input_requirement(self, dataset_dict, model_parameters_dict,
                                boolean_dict):
        key_is_not_required_error = 'key is not required in {}'
        for k, v in dataset_dict.items():
            if k == 'dataset':
                assert isinstance(v, str), f'v must have type = str '
            elif k == 'is_downsampled':
                assert isinstance(v, bool), f'v must have type = bool'
            else:
                raise ValueError(
                    key_is_not_required_error.format('dataset_dict'))
        for k, v in model_parameters_dict.items():
            if k == 'model_name':
                assert isinstance(v, str), f'v must have type = str'
            elif k == 'run_gcn':
                assert isinstance(v, bool), f'v must have type = bool'
            elif k == 'main_epoch':
                assert isinstance(v, int), f'v must have type = int'
            elif k == 'gan_epoch':
                assert isinstance(v, int), f'v must have type = int'
            elif k == 'preserved_edges_percent':
                assert isinstance(v, float), f'v must have type = float'
                assert ((v <= 1) and (
                            v > 0)), 'preserved_edges_percent must be between 0 and 1'
            elif k == 'time_stamp':
                assert isinstance(v, str), f'v must have type = str'
            elif k == 'k_fold_split':
                assert isinstance(v, int), f'v must have type = int'
            elif k == 'device':
                assert isinstance(v, torch.device), f'v must have type = torch.device'
            else:
                raise ValueError(
                    key_is_not_required_error.format('model_parameters_dict'))

        for k, v in boolean_dict.items():

            if k == 'save_dict':
                for t, b in v.items():
                    if t in ['is_saved_each_roc_plot',
                             'is_saved_each_emb_plot',
                             'is_saved_each_performance_plot',
                             'is_saved_each_performance_table']:
                        # for each epoch
                        assert isinstance(b, bool), f'v must have type = bool'
                    elif t in ['is_saved_cv_roc_plot',
                               'is_saved_cv_emb_plot',
                               'is_saved_cv_performance_plot',
                               'is_saved_cv_performance_table']:
                        # for cv
                        assert isinstance(b, bool), f'v must have type = bool'
                    else:
                        raise ValueError(key_is_not_required_error.format(
                            'boolean_dict["save_dict"]'))

            elif k == 'plot_dict':
                for t, b in v.items():
                    if t in ['is_plotted_each_roc',
                             'is_plotted_each_performance',
                             'is_plotted_each_emb',
                             'is_displayed_each_performance_table']:
                        # for each epoch
                        assert isinstance(b, bool), f'v must have type = bool'
                    elif t in ['is_plotted_cv_roc',
                               'is_plotted_cv_performance',
                               'is_plotted_cv_emb',
                               'is_displayed_cv_performance_table']:
                        # for cv
                        assert isinstance(b, bool), f'v must have type = bool'
                    else:
                        raise ValueError(key_is_not_required_error.format(
                            'boolean_dict["plot_dict"]'))
            else:
                raise ValueError(
                    key_is_not_required_error.format('boolean_dict'))

    def init_model(self):
        # =====================
        # ==hyper parameters setup
        # =====================
        self.num_batches = 1
        self.num_epochs = 200

        # =====================
        # ==for gan
        # =====================
        self.gan = gan_model.GAN(self.model_input_data.data,
                                 device=self.model_parameters_dict['device'])
        self.gan.init_gan()

        # =====================
        # ==for Gcn
        # ====================
        self.gcn = gcn_model.GCN(self.model_input_data.data,
                                 preserved_percent=self.model_parameters_dict[
                                     'preserved_edges_percent'],
                                 device=self.model_parameters_dict['device'])

    def prepare_gan_data(self, emb_after_conv1, trainning_selected_min_ind):
        """There are only minority samples that is used as input"""

        trainning_selected_min_y_gcn_emb = self.model_input_data.data.y[
            trainning_selected_min_ind]
        trainning_selected_min_x_gcn_emb = emb_after_conv1[
            trainning_selected_min_ind]

        self.min_class_cora_torch_dataset = CoraTorchDataset(
            trainning_selected_min_x_gcn_emb,
            trainning_selected_min_y_gcn_emb)

        self.batch_size = 10
        self.min_class_data_loader_for_gan = DataLoader(
            self.min_class_cora_torch_dataset, batch_size=self.batch_size,
            shuffle=True)

    def prepare_data_ind(self, fake_data):
        if self.epoch == 0:
            self.model_input_data.set_data(fake_data=fake_data)
            self.model_input_data.set_train_test_data_index()

    def train_and_eval_model_once(self):
        self.gcn.model.train()

        #=====================
        #==gcn train
        #=====================
        self.train_gcn_gan_once()

        # =====================
        # == gcn test
        # =====================
        self.eval_gcn_gan_once()

        # =====================
        # ==Result and performance
        # =====================
        self.display_performance_per_epoch(self.performance_per_epoch)

        # self.train_model_once()
        # self.eval_model_once()

    def run_model_once(self):

        # self.prepare_data_ind()
        self.train_and_eval_model_once()

        self.loss_per_epoch.setdefault('train_loss', []).append(
            self.performance_per_epoch['train_loss_per_epoch'])
        self.loss_per_epoch.setdefault('test_loss', []).append(
            self.performance_per_epoch['test_loss_per_epoch'])

    def run_model(self, ):

        self.run_cv_on_model()

        self.collect_data_for_plotting()
        test_report, test_cm = self.plot_scan_and_loss(
            return_report_stat_for_cv=True)

        # TODO create model_performance_summary
        # avg_auc, avg_acc = test_report.loc['acc/total']['AUC'], \
        #                    test_report.loc['acc/total']['ACC']

        performance_summary_val = [[  # class 0
            test_report.loc['0']['precision'],
            test_report.loc['0']['recall'],
            test_report.loc['0']['f1-score'],
            test_report.loc['0']['support'],
            test_report.loc['0']['predicted'],
            test_report.loc['0']['AUC'],
            test_report.loc['0']['AUC'],
            # class 1
            test_report.loc['1']['precision'],
            test_report.loc['1']['recall'],
            test_report.loc['1']['f1-score'],
            test_report.loc['1']['support'],
            test_report.loc['1']['predicted'],
            test_report.loc['1']['AUC'],
            test_report.loc['1']['AUC'],

            test_report.loc['acc/total']['ACC'],
            test_report.loc['acc/total']['AUC'],

        ]]


        model_performance_summary = pd.DataFrame(performance_summary_val,
                                                 index=self.tuple_row_index,
                                                 columns=self.tuple_col_index)

        # return avg_auc, avg_acc
        return model_performance_summary

    def run_gan_model(self):

        for i in range(self.model_parameters_dict['gan_epoch']):
            print(f'running gan {i}')
            for n_batch, (real_batch, y) in enumerate(
                    self.min_class_data_loader_for_gan):
                self.number_of_sample_per_batch = real_batch.size(0)
                real_data = real_batch

                fake_data = self.gan.generator(
                    gan_model.noise(self.number_of_sample_per_batch).to(
                        self.model_parameters_dict['device']))  # 10, 1433

                error_real, error_fake, d_pred_real, d_pred_fake = \
                    self.gan.train_discriminator(self.gan.d_optimizer,
                                                 real_data,
                                                 fake_data)
                d_error = error_real + error_fake

                fake_data = self.gan.generator(gan_model.noise(
                    self.number_of_sample_per_batch).to(
                    self.model_parameters_dict['device']))

                g_error = self.gan.train_generator(self.gan.g_optimizer,
                                                   fake_data.to(
                                                       self.model_parameters_dict[
                                                           'device']))


                dict_for_plot = {'discriminator_total_loss': d_error,
                                 'generator_loss': g_error,
                                 'discriminator_real_loss': error_real,
                                 'discriminator_fake_loss': error_fake
                                 }
                self.plot_gan_performance.collect_hist_using_list_of_name(
                    dict_for_plot=dict_for_plot)
                self.display_gan_performance_per_main_epoch(d_error, g_error)


    def get_result_per_epoch(self, logits, y_true, is_train=None):
        assert isinstance(is_train, bool), f'is_train must have type = bool '

        y_pred_per_epoch = {}
        y_score_per_epoch = {}
        y_true_per_epoch = {}
        accs_per_epoch = {}
        aucs_per_epoch = {}

        # TODO figure out correct index
        if is_train :
            name_and_mask_dict = {
                'trainning_select_minfake_minreal_majreal_ind': self.model_input_data.trainning_select_minfake_minreal_majreal_ind,
                'trainning_select_minreal_majreal_ind': self.model_input_data.trainning_select_minreal_majreal_ind,
                'trainning_select_min_real_ind': self.model_input_data.trainning_select_min_real_ind,
                'trainning_select_maj_real_ind': self.model_input_data.trainning_select_maj_real_ind,
                'select_min_fake_ind': self.model_input_data.select_min_fake_ind
            }
            for name, mask in name_and_mask_dict.items():

                pred = logits[mask].max(1)[1]
                y_true_mask = y_true[mask]
                y_score = logits[mask]

                y_pred_per_epoch.setdefault(name, pred)
                y_score_per_epoch.setdefault(name, y_score)
                y_true_per_epoch.setdefault(name, y_true_mask)

                acc = pred.eq(y_true_mask).sum().item() / mask.shape[0]
                accs_per_epoch.setdefault(name, acc)
                self.accs_hist_dict.setdefault(name, []).append(acc)

                if name in ['trainning_select_minfake_minreal_majreal_ind',
                            'test_select_minreal_majreal_ind']:
                    auc = get_total_roc_auc_score(
                        y_true_mask.cpu().detach().numpy(), y_score.cpu().detach().numpy())
                    auc = torch.tensor(auc)
                    aucs_per_epoch.setdefault(name, auc)
                    self.aucs_hist_dict.setdefault(name, []).append(auc)

        else:
            name_and_mask_dict = {
                'test_select_minreal_majreal_ind': self.model_input_data.test_select_minreal_majreal_ind,
                'test_select_min_real_ind': self.model_input_data.test_select_min_real_ind,
                'test_select_maj_real_ind': self.model_input_data.test_select_maj_real_ind,
            }

            for name, mask in name_and_mask_dict.items():

                pred = logits[mask].max(1)[1]
                y_true_mask = y_true[mask]
                y_score = logits[
                    mask]

                y_pred_per_epoch.setdefault(name, pred)
                y_score_per_epoch.setdefault(name, y_score)
                y_true_per_epoch.setdefault(name, y_true_mask)

                acc = pred.eq(y_true_mask).sum().item() / mask.shape[0]
                accs_per_epoch.setdefault(name, acc)
                self.accs_hist_dict.setdefault(name, []).append(acc)

                if name in ['trainning_select_minreal_majreal_ind',
                            'test_select_minreal_majreal_ind']:
                    auc = get_total_roc_auc_score(
                        y_true_mask.cpu().detach().numpy(),
                        y_score.cpu().detach().numpy())
                    auc = torch.tensor(auc)
                    aucs_per_epoch.setdefault(name, auc)
                    self.aucs_hist_dict.setdefault(name, []).append(auc)

        return y_true_per_epoch, y_pred_per_epoch, y_score_per_epoch, accs_per_epoch, aucs_per_epoch

    # def display_performance_per_epoch(self, train_loss, test_loss, accs_per_epoch, aucs_per_epoch):
    def display_performance_per_epoch(self, performance_per_epoch):
        print(f"""
            Epoch: {self.epoch:03d},
            Trainning: (loss: {performance_per_epoch['train_loss_per_epoch']:.4f}, min_real: acc={performance_per_epoch['accs_per_epoch']["trainning_select_min_real_ind"]:.4f} maj: acc={performance_per_epoch['accs_per_epoch']["trainning_select_maj_real_ind"]:.4f})  min_fake: acc={performance_per_epoch['accs_per_epoch']["select_min_fake_ind"]:.4f} 
            Test     : (loss: {performance_per_epoch['test_loss_per_epoch']:.4f} min_real = acc={performance_per_epoch['accs_per_epoch']["test_select_min_real_ind"]:.4f} maj: acc={performance_per_epoch['accs_per_epoch']["test_select_maj_real_ind"]:.4f}) 
            Train: (acc={performance_per_epoch['accs_per_epoch']["trainning_select_minfake_minreal_majreal_ind"]:.4f}, auc={performance_per_epoch['aucs_per_epoch']["trainning_select_minfake_minreal_majreal_ind"]: .4f}, loss={performance_per_epoch['train_loss_per_epoch']:.4f})
            Test: ({performance_per_epoch['accs_per_epoch']["test_select_minreal_majreal_ind"]:.4f}, auc={performance_per_epoch['aucs_per_epoch']["test_select_minreal_majreal_ind"]:4f}, loss={performance_per_epoch['test_loss_per_epoch']:.4f})
        """)


    def collect_data_for_plotting(self):
        def avg(x):
            return x / self.model_parameters_dict['k_fold_split']

        avg_train_loss = avg(self.total_loss['train_loss'])
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
                Average over {self.model_parameters_dict['k_fold_split']}
                Trainning: (loss: {avg_train_loss[-1]:.4f}, min_real: acc={avg_trainning_minreal_acc[-1]:.4f} maj: acc={avg_trainning_majreal_acc[-1]:.4f}  min_fake: acc={avg_trainning_minfake_acc[-1]:.4f})
                Test     : (loss: {avg_test_loss[-1]:.4f} min_real = acc={avg_test_minreal_acc[-1]:.4f} maj: acc={avg_test_majreal_acc[-1]:.4f}) 

                Train: (loss: {avg_train_loss[-1]:.4f}, acc={avg_trainning_acc[-1]:.4f}, auc={avg_trainning_auc[-1]: .4f})
                Test:  (loss: {avg_test_loss[-1]:.4f}, acc ={avg_test_acc[-1]:.4f}, auc={avg_test_auc[-1]:4f})
            """)

        dict_for_plot = {'train_loss': avg_train_loss,
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
        # self.plot_gan_performance.
        self.plot_cv_performance.collect_hist_using_list_of_name(
            dict_for_plot=dict_for_plot)


    def plot_scan_and_loss(self,
                           return_report_stat_for_cv):
        """when save_plot is True, it only svae plot that its status is True"""

        naming_convention_dict = {
            'dataset': self.dataset_dict['dataset'],
            'model_name': self.model_parameters_dict['model_name'],
            'is_downsampled': self.dataset_dict['is_downsampled'],
            'main_epoch': self.model_parameters_dict['main_epoch'],
            'gan_epoch': self.model_parameters_dict['gan_epoch'],
            'time_stamp': self.model_parameters_dict['time_stamp'],
            'preserved_edges_percent': self.model_parameters_dict[
                'preserved_edges_percent']
        }

        # =====================
        # ==plot emb
        # =====================
        save_path = self.save_path
        title = self.create_file_naming_convension(naming_convention_dict,
                                                   title=True)
        gan_performance_file_name = self.create_file_naming_convension(naming_convention_dict,
                                                           gan_performance=True)
        emb_file_name = self.create_file_naming_convension(naming_convention_dict,
                                                           emb=True)
        scan_file_name = self.create_file_naming_convension(naming_convention_dict,
                                                            scan=True)
        train_test_name = self.create_file_naming_convension(naming_convention_dict,
                                                             train_test=True)
        report_train_file = self.create_file_naming_convension(
            naming_convention_dict,
            report=True, is_train=True)
        report_test_file = self.create_file_naming_convension(
            naming_convention_dict,
            report=True,
            is_train=False)

        print(save_path)
        print(title)
        print(gan_performance_file_name)
        print(emb_file_name)
        print(scan_file_name)
        print(train_test_name)
        print(report_train_file)
        print(report_test_file)

        list_of_name = {'discriminator_total_loss': (0, 0),
                        'generator_loss': (0, 0),
                        'discriminator_real_loss': (0, 0),
                        'discriminator_fake_loss': (0, 0),
                        }
        self.plot_gan_performance.plot_using_list_of_name(subplot_size=(1, 1),
                                                         name_and_tuple_dict=list_of_name,
                                                         save_file_name=gan_performance_file_name,
                                                         title=title,
                                                         save_path=save_path)

        if self.boolean_dict['plot_dict']['is_plotted_cv_emb'] is not None:
            emb_dict = self.get_emb2d()
            for i, j in emb_dict.items():
                self.plot_cv_emb.collect_emb(i, j)
            self.plot_cv_emb.plot_all(save_path=save_path, title=emb_file_name)

        list_of_name = {'train_loss': (0, 0),
                        'test_loss': (0, 0),
                        'trainning_select_min_real_ind': (1, 0),
                        'select_min_fake_ind': [(1, 0), (2, 0)],
                        'trainning_select_maj_real_ind': (1, 0),
                        'test_select_min_real_ind': (2, 0),
                        'test_select_maj_real_ind': (2, 0),
                        }
        self.plot_cv_performance.plot_using_list_of_name(subplot_size=(3, 1),
                                                         name_and_tuple_dict=list_of_name,
                                                         save_file_name=scan_file_name,
                                                         title=title,
                                                         save_path=save_path)

        # if plot_train_test:

        list_of_name = {'train_loss': (0, 0),
                        'train_acc': (1, 0),
                        'train_auc': (2, 0),
                        'test_loss': (0, 0),
                        'test_acc': (1, 0),
                        'test_auc': (2, 0),
                        }

        self.plot_cv_performance.plot_using_list_of_name(subplot_size=(3, 1),
                                                         name_and_tuple_dict=list_of_name,
                                                         save_file_name=train_test_name,
                                                         title=title,
                                                         save_path=save_path)

        print('=====train========')
        train_performance = self.report_cv_performance.report_performance(
            self.performance_per_epoch['y_true_per_epoch']['trainning_select_minfake_minreal_majreal_ind'],
            self.performance_per_epoch['y_pred_per_epoch']['trainning_select_minfake_minreal_majreal_ind'],
            self.performance_per_epoch['y_score_per_epoch']['trainning_select_minfake_minreal_majreal_ind'],
            labels=np.unique(self.model_input_data.data.y),
            return_value_for_cv=return_report_stat_for_cv,
            save_path=save_path,
            file_name=report_train_file,
        )
        print('=====test======')
        test_performance = self.report_cv_performance.report_performance(
            self.performance_per_epoch['y_true_per_epoch']['test_select_minreal_majreal_ind'],
            self.performance_per_epoch['y_pred_per_epoch']['test_select_minreal_majreal_ind'],
            self.performance_per_epoch['y_score_per_epoch']['test_select_minreal_majreal_ind'],
            labels=np.unique(self.model_input_data.data.y),
            save_path=save_path,
            file_name=report_test_file,
            return_value_for_cv=return_report_stat_for_cv)


        train_report, train_cm = (
            None, None) if train_performance is None else train_performance
        test_report, test_cm = (
            None, None) if test_performance is None else test_performance

        return test_report, test_cm


    def run_cv_on_model(self):
        skf = StratifiedKFold(n_splits=self.model_parameters_dict['k_fold_split'])

        self.gcn.randomedge_sampler()
        original_y = self.model_input_data.data.y
        for train, test in skf.split(
                np.arange(self.model_input_data.data.x.shape[0]),
                original_y):

            self.model_input_data.trainning_selected_ind = train
            self.model_input_data.test_selected_ind = test

            self.loss_per_epoch = {}
            self.accs_hist_dict = {}
            self.aucs_hist_dict = {}

            for epoch in range(self.model_parameters_dict['main_epoch']):
                self.epoch = epoch
                self.run_model_once()

            self.get_total_loss()
            self.get_total_accs_and_aucs()

    def train_gcn_gan_once(self):
        train_emb_after_conv1 = self.gcn.model(self.gcn.get_dgl_graph(),
                                               self.model_input_data.data.x.to(self.model_parameters_dict['device']),
                                               get_conv1_emb=True)

        # =====================
        # ==gan dataset
        # =====================
        self.prepare_gan_data(
            train_emb_after_conv1,
            self.model_input_data.trainning_selected_min_ind)

        self.run_gan_model()

        # =====================
        # == fake_neg/true_neg/pos = 25%/25%/50%
        # =====================
        fake_data = self.gan.generator(gan_model.noise(
            self.model_input_data.trainning_selected_min_ind.shape[
                0]).to(
            self.model_parameters_dict['device']))  # this will be sent to discriminator 2 too

        minreal_minfake_majreal_x = torch.cat(
            (train_emb_after_conv1, fake_data), 0).to(self.model_parameters_dict['device'])
        minreal_minfake_majreal_y = torch.cat(
            (self.model_input_data.data.y,
             torch.zeros(
                 fake_data.size(0)).type(
                 torch.long)), 0).type(
            torch.long).to(self.model_parameters_dict['device'])

        #=====================
        #==prepare ind
        #=====================

        self.prepare_data_ind(fake_data)

        #=====================
        #==train
        #=====================

        train_emb_after_conv2, train_emb_after_conv2_with_external_input, logits = self.gcn.model(
            self.gcn.get_dgl_graph(),
            train_emb_after_conv1,
            external_input=fake_data.to(
                self.model_parameters_dict['device']),
            run_discriminator=True)
        train_loss_per_epoch = self.gcn.loss_and_step(
            logits[self.model_input_data.trainning_select_minfake_minreal_majreal_ind],
            minreal_minfake_majreal_y[
                self.model_input_data.trainning_select_minfake_minreal_majreal_ind])

        y_true_per_epoch, y_pred_per_epoch, y_score_per_epoch, accs_per_epoch, aucs_per_epoch  = self.get_result_per_epoch(logits, minreal_minfake_majreal_y,is_train=True)


        self.performance_per_epoch.setdefault('y_true_per_epoch', {}).update(
            y_true_per_epoch)
        self.performance_per_epoch.setdefault('y_pred_per_epoch', {}).update(
            y_pred_per_epoch)
        self.performance_per_epoch.setdefault('y_score_per_epoch', {}).update(
            y_score_per_epoch)

        self.performance_per_epoch['train_loss_per_epoch'] = train_loss_per_epoch
        self.performance_per_epoch['emb'] = train_emb_after_conv2_with_external_input

        self.performance_per_epoch.setdefault('accs_per_epoch', {}).update(
            accs_per_epoch)
        self.performance_per_epoch.setdefault('aucs_per_epoch', {}).update(
            aucs_per_epoch)

        # return y_true_per_epoch, y_pred_per_epoch, y_score_per_epoch, accs_per_epoch, aucs_per_epoch

    def eval_gcn_gan_once(self):
        self.gcn.model.eval()

        test_emb_after_conv1, test_emb_after_conv2, logits = self.gcn.model(
            self.gcn.get_dgl_graph(), self.model_input_data.data.x.to(
                self.model_parameters_dict['device']),
            run_all=True)

        test_loss_per_epoch = self.gcn.loss(
            logits[self.model_input_data.test_selected_ind],
            self.model_input_data.data.y[
                self.model_input_data.test_selected_ind].to(
                self.model_parameters_dict['device']))

        y_true_per_epoch, y_pred_per_epoch, y_score_per_epoch, accs_per_epoch, aucs_per_epoch = self.get_result_per_epoch(
            logits, self.model_input_data.data.y, is_train=False)

        self.performance_per_epoch.setdefault('y_true_per_epoch', {}).update(
            y_true_per_epoch)
        self.performance_per_epoch.setdefault('y_pred_per_epoch', {}).update(
            y_pred_per_epoch)
        self.performance_per_epoch.setdefault('y_score_per_epoch', {}).update(
            y_score_per_epoch)


        self.performance_per_epoch['test_loss_per_epoch'] = test_loss_per_epoch
        self.performance_per_epoch.setdefault('accs_per_epoch', {}).update(
            accs_per_epoch)
        self.performance_per_epoch.setdefault('aucs_per_epoch', {}).update(
            aucs_per_epoch)

        # return y_true_per_epoch, y_pred_per_epoch, y_score_per_epoch, accs_per_epoch, aucs_per_epoch

    def get_emb2d(self):
        emb_dict={}
        emb_2d = apply_tsne_on_emb(
            self.performance_per_epoch['emb'].cpu().detach().numpy(),
            run_gcn=self.model_parameters_dict['run_gcn'])
        emb_dict.setdefault('test', emb_2d[self.model_input_data.test_selected_ind])
        emb_dict.setdefault('train', emb_2d[self.model_input_data.trainning_selected_ind])
        emb_dict.setdefault('min_real', np.concatenate((emb_2d[
                                                            self.model_input_data.trainning_select_min_real_ind.cpu().detach().numpy()],
                                                        emb_2d[
                                                            self.model_input_data.test_select_min_real_ind.cpu().detach().numpy()]),
                                                       axis=0))
        emb_dict.setdefault('min_fake', emb_2d[self.model_input_data.select_min_fake_ind])
        emb_dict.setdefault('maj', np.concatenate((emb_2d[
                                                       self.model_input_data.trainning_select_maj_real_ind.cpu().detach().numpy()],
                                                   emb_2d[
                                                       self.model_input_data.test_select_maj_real_ind.cpu().detach().numpy()]),
                                                  0))
        return emb_dict

    def display_gan_performance_per_main_epoch(self, d_error, g_error):
        print(f"""discriminator_loss: {d_error}, generator_loss: {g_error}""")

    def create_multi_level_row_index(self):

        tuple_row_index = [(self.dataset_dict['dataset'],
                            self.model_parameters_dict['model_name'],
                            f"percent={self.model_parameters_dict['preserved_edges_percent']}",
                            f"main_epoch={self.model_parameters_dict['main_epoch']}",
                            f"gan_epoch={self.model_parameters_dict['gan_epoch']}",
                            )]

        self.tuple_row_index = pd.MultiIndex.from_tuples(tuple_row_index)

    def create_multi_level_col_index(self):

        tuple_col_index = (
            # 16 columsn
            ('class0', 'precision'),
            ('class0', 'recall'),
            ('class0', 'f1'),
            ('class0', 'support'),
            ('class0', 'predicted'),
            ('class0', 'Acc'),
            ('class0', 'AUC'),

            ('class1', 'precision'),
            ('class1', 'recall'),
            ('class1', 'f1'),
            ('class1', 'support'),
            ('class1', 'predicted'),
            ('class1', 'Acc'),
            ('class1', 'AUC'),

            ('total', 'total_accs'),
            ('total', 'total_aucs'),
        )

        self.tuple_col_index = pd.MultiIndex.from_tuples(tuple_col_index)


if __name__ == '__main__':

    np.random.seed(111)
    torch.manual_seed(111)


    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    assert sum([args.run_gcn, args.run_gcn_gan]) ==1, ''
    if args.run_gcn:
        model_name = 'run_gcn'
    elif args.run_gcn_gan:
        model_name = 'train_model'
    else:
        raise ValueError('')

    # run_gcn = True if args.run_gcn == 'run_gcn' else False
    time_stamp = time.strftime("%Y%m%d-%H%M%S")

    dataset_dict = {
        'dataset': args.dataset,  # name of data
        'is_downsampled': args.is_downsampled,
    }
    model_parameters_dict = {
        'model_name': model_name,
        'run_gcn': args.run_gcn,
        'main_epoch': args.main_epoch,
        'gan_epoch': args.gan_epoch,
        'preserved_edges_percent': args.preserved_edges_percent,
        'time_stamp': time_stamp,
        'k_fold_split': args.k_fold_split,
        'device': device,
    }

    #=====================
    #==each
    #=====================

    # args.is_saved_each_roc_plot= True
    # args.is_saved_each_emb_plot= True
    # args.is_saved_each_performance_plot= True
    # args.is_saved_each_performance_table= True
    # args.is_plotted_each_roc= True
    # args.is_plotted_each_emb= True
    # args.is_plotted_each_performance= True
    # args.is_displayed_each_performance_table= True


    #=====================
    #==cv
    #=====================

    args.is_plotted_cv_roc= True
    args.is_plotted_cv_emb= True
    args.is_plotted_cv_performance= True
    args.is_displayed_cv_performance_table= True
    args.is_downsampled = True

    # args.is_saved_cv_roc_plot= True
    # args.is_saved_cv_emb_plot= True
    # args.is_saved_cv_performance_plot= True
    # args.is_saved_cv_performance_table= True

    boolean_dict = {
        'save_dict': {
            'is_saved_each_roc_plot': args.is_saved_each_roc_plot,
            'is_saved_each_emb_plot': args.is_saved_each_emb_plot,
            'is_saved_each_performance_plot': args.is_saved_each_performance_plot,
            'is_saved_each_performance_table': args.is_saved_each_performance_table,
            'is_saved_cv_roc_plot': args.is_saved_cv_roc_plot,
            'is_saved_cv_emb_plot': args.is_saved_cv_emb_plot,
            'is_saved_cv_performance_plot': args.is_saved_cv_performance_plot,
            'is_saved_cv_performance_table': args.is_saved_cv_performance_table
        },
        'plot_dict': {
            'is_plotted_each_roc': args.is_plotted_each_roc,
            'is_plotted_each_emb': args.is_plotted_each_emb,
            'is_plotted_each_performance': args.is_plotted_each_performance,
            'is_displayed_each_performance_table': args.is_displayed_each_performance_table,
            'is_plotted_cv_roc': args.is_plotted_cv_roc,
            'is_plotted_cv_emb': args.is_plotted_cv_emb,
            'is_plotted_cv_performance': args.is_plotted_cv_performance,
            'is_displayed_cv_performance_table': args.is_displayed_cv_performance_table
        }
    }

    # my_new_model = MyNewModel(dataset_dict.copy(), model_parameters_dict.copy(),
    #                           boolean_dict.copy())
    gcn_gan = GCNGAN(dataset_dict, model_parameters_dict, boolean_dict)

    avg_auc, avg_acc = gcn_gan.run_model()

