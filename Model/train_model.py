import inspect
import os
import sys
import time

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
from src.Preparation.Data import preparing_cora_for_new_purposed_model
# from Plot import PlotClass
from src.Visualization import PlotClass
from src.Evaluation import get_total_roc_auc_score
from src.Preprocessing import ModelInputData
from src.Evaluation import report_performance

log = Logging.Logger(name='log_for_train_model_file')


# warnings.simplefilter("error")
# warnings.simplefilter("ignore", DeprecationWarning)
# warnings.simplefilter("ignore", UserWarning)

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


class MyNewModel:
    # def __init__(self, data, dataset, dataloader):
    def __init__(self,dataset, k_fold_split=3, isLog=False, run_gcn_only=False):
        assert isinstance(dataset, str), ' please specify dataset '
        self.k_fold_split = k_fold_split
        self.time_stamp = time.strftime("%Y%m%d-%H%M%S")

        self.plot_class = PlotClass(save_status=args.save_plot)

        # self.data = data
        self.log = isLog
        self.dataset = dataset
        self.model_input_data = ModelInputData(self.dataset)
        self.data = self.model_input_data.data
        self.init_my_new_model()
        self.run_gcn_only = run_gcn_only
        self.total_accs_dict = {}
        self.total_aucs_dict = {}
        self.total_loss = {}


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

    def run_gan_components_of_new_model(self, num_gan_epoch):
        if self.log:
            log.info('in run_gan_components_of_new_model...')

        if num_gan_epoch:
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

        # for name, mask in self.data('trainning_select_minreal_majreal_ind',
        #                             'test_select_minfake_minreal_majreal_ind',
        #                             'trainning_select_min_real_ind',
        #                             'trainning_select_maj_real_ind',
        #                             'test_select_min_real_ind',
        #                             'test_select_maj_real_ind',
        #                             ):
        for name, mask in name_and_mask_dict.items():

            pred = logits[mask].max(1)[1]
            y_true_mask = y_true[mask]
            y_score = logits[mask].detach().numpy()

            y_pred_dict.setdefault(name, pred.detach().numpy())
            y_score_dict.setdefault(name, y_score)
            y_true_dict.setdefault(name, y_true_mask.detach().numpy())

            acc = pred.eq(y_true_mask).sum().item() / mask.shape[0]
            accs_dict.setdefault(name, acc)

            if name in ['trainning_select_minreal_majreal_ind',
                        'test_select_minreal_majreal_ind']:
                auc = get_total_roc_auc_score(y_true_mask, y_score)
                aucs_dict.setdefault(name, auc)

        return accs_dict, aucs_dict, y_pred_dict, y_score_dict, y_true_dict

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

    def run_my_new_model_once(self, epoch):
        if self.log:
            log.info('in run_my_new_model_once..')

        # convert edge_index to adj
        # TODO does this have any connection to otehr part of the model? maybe I just forget to connect it to other components
        self.data.edge_index = randomedge_sampler(self.data.edge_index, 1,
                                                  isLog=self.log)

        y_pred_dict = {}
        y_score_dict = {}
        y_true_dict = {}
        accs_dict = {}
        aucs_dict = {}

        if not self.run_gcn_only:

            self.gcn.model.train()

            emb_after_conv1 = self.gcn.model(self.gcn.get_dgl_graph(),
                                             self.data.x,
                                             get_conv1_emb=True)

            # =====================
            # ==gan dataset
            # =====================
            self.prepare_gan_trainning_dataset(
                emb_after_conv1,
                self.model_input_data.trainning_selected_min_ind)

            self.run_gan_components_of_new_model(num_gan_epoch=args.num_gan_epoch)

            # =====================
            # == fake_neg/true_neg/pos = 25%/25%/50%
            # =====================
            fake_data = self.gan.generator(gan_model.noise(
                self.model_input_data.trainning_selected_min_ind.shape[
                    0]))  # this will be sent to discriminator 2 too

            minreal_minfake_majreal_x = torch.cat(
                (emb_after_conv1, fake_data), 0)
            minreal_minfake_majreal_y = torch.cat((torch.tensor(self.data.y),
                                                   torch.zeros(
                                                       fake_data.size(0)).type(
                                                       torch.int)), 0).type(
                torch.long)

            if epoch == 0:
                self.model_input_data.set_data(fake_data=fake_data)
                self.model_input_data.set_train_test_data_index()
            self.prepare_data_ind()


            emb_after_conv2, logits = self.gcn.model(self.gcn.get_dgl_graph(),
                                                     emb_after_conv1,
                                                     external_input=fake_data,
                                                     run_discriminator=True)
            trainning_loss = self.gcn.loss_and_step(
                logits[self.data.trainning_select_minfake_minreal_majreal_ind],
                minreal_minfake_majreal_y[
                    self.data.trainning_select_minfake_minreal_majreal_ind])


            y_true = minreal_minfake_majreal_y

            name_and_mask_dict = {
                'trainning_select_minfake_minreal_majreal_ind': self.data.trainning_select_minfake_minreal_majreal_ind,
                'trainning_select_min_real_ind': self.data.trainning_select_min_real_ind,
                'trainning_select_maj_real_ind': self.data.trainning_select_maj_real_ind,
                'select_min_fake_ind': self.data.select_min_fake_ind,
            }

            for name, mask in name_and_mask_dict.items():

            # for name, mask in self.data(
            #         'trainning_select_minfake_minreal_majreal_ind',
            #         'trainning_select_min_real_ind',
            #         'trainning_select_maj_real_ind',
            #         'select_min_fake_ind'
            #         ):

                pred = logits[mask].max(1)[1]
                y_true_mask = y_true[mask]
                y_score = logits[mask].detach().numpy()

                y_pred_dict.setdefault(name, pred.detach().numpy())
                y_score_dict.setdefault(name, y_score)
                y_true_dict.setdefault(name, y_true_mask.detach().numpy())

                acc = pred.eq(y_true_mask).sum().item() / mask.shape[0]
                accs_dict.setdefault(name, acc)
                self.accs_hist_dict.setdefault(name, []).append(acc)

                if name in ['trainning_select_minfake_minreal_majreal_ind',
                            'test_select_minreal_majreal_ind']:
                    auc = get_total_roc_auc_score(y_true_mask, y_score)
                    aucs_dict.setdefault(name, auc)
                    # accs_hist_dict.setdefault(name, []).append(acc)
                    self.aucs_hist_dict.setdefault(name, []).append(auc)

            # =====================
            # == gcn test
            # =====================
            self.gcn.model.eval()

            logits = self.gcn.model(
                self.gcn.get_dgl_graph(), self.data.x,
                run_all=True)

            test_loss = self.gcn.loss(
                logits[self.data.test_selected_ind],
                torch.tensor(self.data.y).type(torch.long)[
                    self.data.test_selected_ind])

            # accs_dict, aucs_dict, y_pred_dict, y_score_dict, y_true_dict = self.collect_performance(
            #     logits, torch.tensor(self.data.y).type(torch.long), is_test=True)


            name_and_mask_dict = {
                'test_select_minreal_majreal_ind': self.data.test_select_minreal_majreal_ind,
                'test_select_min_real_ind': self.data.test_select_min_real_ind,
                'test_select_maj_real_ind': self.data.test_select_maj_real_ind,
            }

            for name, mask in name_and_mask_dict.items():

            # for name, mask in self.data(
            #         'test_select_minreal_majreal_ind',
            #         'test_select_min_real_ind',
            #         'test_select_maj_real_ind',
            #                             ):

                pred = logits[mask].max(1)[1]
                y_true_mask = y_true[mask]
                y_score = logits[mask].detach().numpy()

                y_pred_dict.setdefault(name, pred.detach().numpy())
                y_score_dict.setdefault(name, y_score)
                y_true_dict.setdefault(name, y_true_mask.detach().numpy())


                acc = pred.eq(y_true_mask).sum().item() / mask.shape[0]
                accs_dict.setdefault(name, acc)
                self.accs_hist_dict.setdefault(name, []).append(acc)

                if name in ['trainning_select_minreal_majreal_ind',
                            'test_select_minreal_majreal_ind']:
                    auc = get_total_roc_auc_score(y_true_mask, y_score)
                    aucs_dict.setdefault(name, auc)
                    # accs_hist_dict.setdefault(name, []).append(acc)
                    self.aucs_hist_dict.setdefault(name, []).append(auc)

        else:

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

            self.gcn.model.train()

            logits = self.gcn.model(self.gcn.get_dgl_graph(),
                                    self.data.x,
                                    run_all=True)

            trainning_loss = self.gcn.loss_and_step(
                logits[self.model_input_data.trainning_selected_ind],
                torch.tensor(self.data.y).type(torch.long)[
                    self.model_input_data.trainning_selected_ind])

            # =====================
            # == gcn test
            # =====================
            self.gcn.model.eval()

            logits = self.gcn.model(
                self.gcn.get_dgl_graph(), self.data.x,
                run_all=True)

            test_loss = self.gcn.loss_and_step(
                logits[self.model_input_data.test_selected_ind],
                torch.tensor(self.data.y).type(torch.long)[
                    self.model_input_data.test_selected_ind])
            # accs_dict, aucs_dict, y_pred_dict, y_score_dict, y_true_dict = self.collect_performance(
            #     logits, self.data.y)


            y_true = torch.tensor(self.data.y).type(torch.long)

            name_and_mask_dict = {
                'trainning_select_minreal_majreal_ind': self.data.trainning_select_minreal_majreal_ind,
                'test_select_minreal_majreal_ind': self.data.test_select_minreal_majreal_ind,
                'trainning_select_min_real_ind': self.data.trainning_select_min_real_ind,
                'trainning_select_maj_real_ind': self.data.trainning_select_maj_real_ind,
                'test_select_min_real_ind': self.data.test_select_min_real_ind,
                'test_select_maj_real_ind': self.data.test_select_maj_real_ind
            }

            # for name, mask in self.data('trainning_select_minreal_majreal_ind',
            #                             'test_select_minreal_majreal_ind',
            #                             'trainning_select_min_real_ind',
            #                             'trainning_select_maj_real_ind',
            #                             'test_select_min_real_ind',
            #                             'test_select_maj_real_ind',
            #                             ):

            for name, mask in  name_and_mask_dict.items():

                pred = logits[mask].max(1)[1]
                y_true_mask = y_true[mask]
                y_score = logits[mask].detach().numpy()

                y_pred_dict.setdefault(name, pred.detach().numpy())
                y_score_dict.setdefault(name, y_score)
                y_true_dict.setdefault(name, y_true_mask.detach().numpy())

                acc = pred.eq(y_true_mask).sum().item() / mask.shape[0]
                accs_dict.setdefault(name, acc)
                self.accs_hist_dict.setdefault(name, []).append(acc)

                if name in ['trainning_select_minreal_majreal_ind',
                            'test_select_minreal_majreal_ind']:
                    auc = get_total_roc_auc_score(y_true_mask, y_score)
                    aucs_dict.setdefault(name, auc)
                    self.aucs_hist_dict.setdefault(name, []).append(auc)

        if not self.run_gcn_only:
            print(f"""
                Epoch: {epoch:03d},
                Trainning: (loss: {trainning_loss:.4f}, min_real: acc={accs_dict["trainning_select_min_real_ind"]:.4f} maj: acc={accs_dict["trainning_select_maj_real_ind"]:.4f})  min_fake: acc={accs_dict["select_min_fake_ind"]:.4f}
                Test     : (loss: {test_loss:.4f} min_real = acc={accs_dict["test_select_min_real_ind"]:.4f} maj: acc={accs_dict["test_select_maj_real_ind"]:.4f}) 

                Train: (acc={accs_dict["trainning_select_minfake_minreal_majreal_ind"]:.4f}, auc={aucs_dict["trainning_select_minfake_minreal_majreal_ind"]: .4f}, loss={trainning_loss:.4f})
                Test: ({accs_dict["test_select_minreal_majreal_ind"]:.4f}, auc={aucs_dict["test_select_minreal_majreal_ind"]:4f}, loss={test_loss:.4f})
            """)

            name_and_val_dict = {'train_loss': trainning_loss,
                                 'train_acc': accs_dict['trainning_select_minfake_minreal_majreal_ind'],
                                 'trainning_select_min_real_ind': accs_dict['trainning_select_min_real_ind'],
                                 'trainning_select_maj_real_ind': accs_dict['trainning_select_maj_real_ind'],
                                 'train_auc': aucs_dict['trainning_select_minfake_minreal_majreal_ind'],
                                 'test_loss': test_loss,
                                 'test_acc': accs_dict['test_select_minreal_majreal_ind'],
                                'test_auc': aucs_dict['test_select_minreal_majreal_ind'],
                                 'test_select_min_real_ind': accs_dict['test_select_min_real_ind'],
                                 'test_select_maj_real_ind': accs_dict['test_select_maj_real_ind'],
                                 'select_min_fake_ind': accs_dict['select_min_fake_ind']
                                 }
            self.plot_class.collect_hist_using_list_of_name(
                name_and_val_dict=name_and_val_dict)

        else:
            print(f"""
            Epoch: {epoch:03d},
            Trainning: (loss: {trainning_loss:.4f}, min_real: acc={accs_dict["trainning_select_min_real_ind"]:.4f} maj: acc={accs_dict["trainning_select_maj_real_ind"]:.4f})
            Test     : (loss: {test_loss:.4f} min_real: acc={accs_dict["test_select_min_real_ind"]:.4f} maj: acc={accs_dict["test_select_maj_real_ind"]:.4f}) 

            Train: (acc={accs_dict["trainning_select_minreal_majreal_ind"]:.4f}, auc={aucs_dict["trainning_select_minreal_majreal_ind"]: .4f}, loss={trainning_loss:.4f})
            Test: ({accs_dict["test_select_minreal_majreal_ind"]:.4f}, auc={aucs_dict["test_select_minreal_majreal_ind"]:4f}, loss={test_loss:.4f})
        """)

            name_and_val_dict = {'train_loss': trainning_loss,
                                 'train_acc': accs_dict['trainning_select_minreal_majreal_ind'],
                                 'trainning_select_min_real_ind': accs_dict['trainning_select_min_real_ind'],
                                 'trainning_select_maj_real_ind': accs_dict['trainning_select_maj_real_ind'],
                                 'train_auc': aucs_dict['trainning_select_minreal_majreal_ind'],
                                 'test_loss': test_loss,
                                 'test_acc': accs_dict['test_select_minreal_majreal_ind'],
                                 'test_auc': aucs_dict['test_select_minreal_majreal_ind'],
                                 'test_select_min_real_ind': accs_dict['test_select_min_real_ind'],
                                 'test_select_maj_real_ind': accs_dict['test_select_maj_real_ind'],
                                 }
            self.plot_class.collect_hist_using_list_of_name(
                name_and_val_dict=name_and_val_dict)

        self.loss_per_epoch.setdefault('trainning_loss', []).append(trainning_loss.detach().numpy())
        self.loss_per_epoch.setdefault('test_loss', []).append(test_loss.detach().numpy())

        self.y_true_dict = y_true_dict
        self.y_pred_dict = y_pred_dict
        self.y_score_dict = y_score_dict
        self.accs_dict = y_score_dict
        self.aucs_dict = aucs_dict

    def collect_data_for_plotting(self):
        def avg(x):
            return x/self.k_fold_split

        if not self.run_gcn_only:

            # TODO how to avg over list => sum element wise then divide by k_fold_split?
            avg_trainning_loss = avg(self.total_loss['trainning_loss'])
            avg_test_loss = avg(self.total_loss['test_loss'])
            avg_trainning_minfake_acc = avg(self.total_accs_dict['select_min_fake_ind'])
            avg_trainning_minreal_acc = avg(self.total_accs_dict['trainning_select_min_real_ind'])
            avg_trainning_majreal_acc = avg(self.total_accs_dict['trainning_select_maj_real_ind'])
            avg_trainning_acc = avg(self.total_accs_dict['trainning_select_minfake_minreal_majreal_ind'])
            avg_trainning_auc = avg(self.total_aucs_dict['trainning_select_minfake_minreal_majreal_ind'])

            avg_test_minreal_acc = avg(self.total_accs_dict['test_select_min_real_ind'])
            avg_test_majreal_acc = avg(self.total_accs_dict['test_select_maj_real_ind'])
            avg_test_acc = avg(self.total_accs_dict['test_select_minreal_majreal_ind'])
            avg_test_auc = avg(self.total_aucs_dict['test_select_minreal_majreal_ind'])

            print(f"""
                Average over {self.k_fold_split}
                Trainning: (loss: {avg_trainning_loss[-1]:.4f}, min_real: acc={avg_trainning_minreal_acc[-1]:.4f} maj: acc={avg_trainning_majreal_acc[-1]:.4f}  min_fake: acc={avg_trainning_minfake_acc[-1]:.4f})
                Test     : (loss: {avg_test_loss[-1]:.4f} min_real = acc={avg_test_minreal_acc[-1]:.4f} maj: acc={avg_test_majreal_acc[-1]:.4f}) 

                Train: (loss: {avg_trainning_loss[-1]:.4f}, acc={avg_trainning_acc[-1]:.4f}, auc={avg_trainning_auc[-1]: .4f})
                Test:  (loss: {avg_test_loss[-1]:.4f}, acc ={avg_test_acc[-1]:.4f}, auc={avg_test_auc[-1]:4f})
            """)


            self.plot_class.hist = {}
            name_and_val_dict = {'train_loss': avg_trainning_loss,
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
            self.plot_class.collect_hist_using_list_of_name(name_and_val_dict=name_and_val_dict)

        else:


            avg_trainning_loss = avg(self.total_loss['trainning_loss'])
            avg_test_loss = avg(self.total_loss['test_loss'])
            avg_trainning_minreal_acc = avg(self.total_accs_dict['trainning_select_min_real_ind'])
            avg_trainning_majreal_acc = avg(self.total_accs_dict['trainning_select_maj_real_ind'])
            avg_trainning_acc = avg(self.total_accs_dict['trainning_select_minreal_majreal_ind'])
            avg_trainning_auc = avg(self.total_aucs_dict['trainning_select_minreal_majreal_ind'])

            avg_test_minreal_acc = avg(self.total_accs_dict['test_select_min_real_ind'])
            avg_test_majreal_acc = avg(self.total_accs_dict['test_select_maj_real_ind'])
            avg_test_acc = avg(self.total_accs_dict['test_select_minreal_majreal_ind'])
            avg_test_auc = avg(self.total_aucs_dict['test_select_minreal_majreal_ind'])

            print(f"""
                Average over {self.k_fold_split}
                Trainning: (loss: {avg_trainning_loss[-1]:.4f}, min_real: acc={avg_trainning_minreal_acc[-1]:.4f} maj: acc={avg_trainning_majreal_acc[-1]:.4f})
                Test     : (loss: {avg_test_loss[-1]:.4f} min_real = acc={avg_test_minreal_acc[-1]:.4f} maj: acc={avg_test_majreal_acc[-1]:.4f}) 

                Train: (loss: {avg_trainning_loss[-1]:.4f}, acc={avg_trainning_acc[-1]:.4f}, auc={avg_trainning_auc[-1]: .4f})
                Test:  (loss: {avg_test_loss[-1]:.4f}, acc ={avg_test_acc[-1]:.4f}, auc={avg_test_auc[-1]:4f})
            """)

            self.plot_class.hist = {}
            name_and_val_dict = {'train_loss': avg_trainning_loss,
                                 'train_acc': avg_trainning_acc,
                                 'trainning_select_min_real_ind': avg_trainning_minreal_acc,
                                 'trainning_select_maj_real_ind': avg_trainning_majreal_acc,
                                 'train_auc': avg_trainning_auc,
                                 'test_loss': avg_test_loss,
                                 'test_acc': avg_test_acc,
                                 'test_auc': avg_test_auc,
                                 'test_select_min_real_ind': avg_test_minreal_acc, 'test_select_maj_real_ind': avg_test_majreal_acc,
                                 }
            self.plot_class.collect_hist_using_list_of_name(
                name_and_val_dict=name_and_val_dict)
    def get_total_loss(self):
        if 'trainning_loss' not in self.total_loss:
            self.total_loss['trainning_loss'] = np.array(
                self.loss_per_epoch['trainning_loss'])
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

    def run_my_new_model(self, ):

        # self.model_input_data.cora_prepare_ind_for_trainning_and_test_set()
        # self.model_input_data.citeseer_prepare_ind_for_trainning_and_test_set()
        # TODO cross validation

        skf = StratifiedKFold(n_splits=self.k_fold_split)

        original_y = self.data.y
        for train, test in skf.split(np.arange(self.data.x.shape[0]), original_y):
            self.model_input_data.trainning_selected_ind = train
            self.model_input_data.test_selected_ind = test

            self.loss_per_epoch = {}
            self.accs_hist_dict = {}
            self.aucs_hist_dict = {}

            # for epoch in range(self.num_epochs):
            # for epoch in range(2):
            # for epoch in range(30):
            for epoch in range(60):
                self.run_my_new_model_once(
                    epoch)

            self.get_total_loss()
            self.get_total_accs_and_aucs()


            self.plot_scan_and_loss( plot_scan=False,
                                         plot_train_test=False,
                                         plot_roc=False,
                                         save_file=args.save_file,
                                         save_plot=args.save_plot,
                                         display_report=True)


        # TODO here>> avg over these total dict

        self.collect_data_for_plotting()

        self.plot_scan_and_loss( plot_scan=True,
                                 plot_train_test=True,
                                     plot_roc=True,
                                     save_file=args.save_cv_file,
                                     save_plot=args.save_cv_plot,
                                     display_report=True)

    def plot_scan_and_loss(self,
                           plot_scan=False,
                           plot_train_test=False,
                           plot_roc=False,
                           save_file=False,
                           save_plot=False,
                           display_report=False,
                           return_report_stat_for_cv=False):
        """when save_plot is True, it only svae plot that its status is True"""
        self.plot_class.save_status = save_plot
        if not self.run_gcn_only:

            if plot_scan:

                list_of_name = {'train_loss':(0,0),
                                'test_loss':(0,0),
                                'trainning_select_min_real_ind':(1,0),
                                'select_min_fake_ind':[(1,0),(2,0)],
                                'trainning_select_maj_real_ind': (1,0),
                                'test_select_min_real_ind':(2,0),
                                'test_select_maj_real_ind':(2,0),
                }

                file_name = f'from_train_model_scan = True_{self.time_stamp}'
                self.plot_class.plot_using_list_of_name(subplot_size=(3,1),name_and_tuple_dict=list_of_name, save_file_name=file_name, title=f'train_model_{self.time_stamp}')

            if plot_train_test:
                list_of_name = {'train_loss':(0,0),
                                'train_acc':(1,0),
                                'train_auc':(2,0),
                                'test_loss':(0,0),
                                'test_acc': (1,0),
                                'test_auc':(2,0),
                                }

                file_name = f'from_train_model_{self.time_stamp}'
                self.plot_class.plot_using_list_of_name(subplot_size=(3,1),name_and_tuple_dict=list_of_name, save_file_name=file_name, title=f'train_model_{self.time_stamp}')

            print('=====train========')
            report_train_file = f'train_model_train_{self.time_stamp}'
            report_test_file = f'train_model_test_{self.time_stamp}'
            train_report = report_performance(
                self.y_true_dict['trainning_select_minfake_minreal_majreal_ind'],
                self.y_pred_dict['trainning_select_minfake_minreal_majreal_ind'],
                self.y_score_dict['trainning_select_minfake_minreal_majreal_ind'],
                labels=np.unique(self.data.y), verbose=display_report,
                plot=plot_roc,
                save_status=save_file,
                save_path=f'C:\\Users\\Anak\\PycharmProjects\\AdaptiveGraphStructureEmbedding\\Output\\Report\\{self.dataset}\\train_model\\',
                file_name=report_train_file,
                return_value_for_cv=return_report_stat_for_cv
            )
            print('=====test======')
            test_report = report_performance(self.y_true_dict['test_select_minreal_majreal_ind'],
                               self.y_pred_dict['test_select_minreal_majreal_ind'],
                               self.y_score_dict['test_select_minreal_majreal_ind'],
                               labels=np.unique(self.data.y), verbose=display_report,
                               plot=plot_roc,
                               save_status=save_file,
                               save_path=f'C:\\Users\\Anak\\PycharmProjects\\AdaptiveGraphStructureEmbedding\\Output\\Report\\{self.dataset}\\train_model\\',
                               file_name=report_test_file,
                                return_value_for_cv = return_report_stat_for_cv)

        else:
            if plot_scan:

                list_of_name = {'train_loss':(0,0),
                                'test_loss':(0,0),
                                'trainning_select_min_real_ind':(1,0),
                                'trainning_select_maj_real_ind': (1,0),
                                'test_select_min_real_ind':(2,0),
                                'test_select_maj_real_ind':(2,0),
                                }

                file_name = f'from_train_model_scan=True_{self.time_stamp}'
                self.plot_class.plot_using_list_of_name(subplot_size=(3,1),name_and_tuple_dict=list_of_name, save_file_name=file_name, title=f'run_gcn_{self.time_stamp}')

            if plot_train_test:

                list_of_name = {'train_loss':(0,0),
                                'train_acc':(1,0),
                                'train_auc':(2,0),
                                'test_loss':(0,0),
                                'test_acc': (1,0),
                                'test_auc':(2,0),
                                }

                file_name = f'from_run_gcn_{self.time_stamp}'
                self.plot_class.plot_using_list_of_name(subplot_size=(3,1),name_and_tuple_dict=list_of_name, save_file_name=file_name, title=f'train_model_{self.time_stamp}')

            print('=====train========')
            report_train_file = f'run_gcn_train_{self.time_stamp}'
            report_test_file = f'run_gcn_test_{self.time_stamp}'
            train_report = report_performance(
                self.y_true_dict['trainning_select_minreal_majreal_ind'],
                self.y_pred_dict['trainning_select_minreal_majreal_ind'],
                self.y_score_dict['trainning_select_minreal_majreal_ind'],
                labels=np.unique(self.data.y), verbose=display_report,
                plot=plot_roc,
                save_status=save_file,
                save_path=f'C:\\Users\\Anak\\PycharmProjects\\AdaptiveGraphStructureEmbedding\\Output\\Report\\{self.dataset}\\train_model\\',
                file_name=report_train_file,
                return_value_for_cv=return_report_stat_for_cv)
            print('=====test======')
            test_report = report_performance(
                self.y_true_dict['test_select_minreal_majreal_ind'],
                               self.y_pred_dict['test_select_minreal_majreal_ind'],
                               self.y_score_dict['test_select_minreal_majreal_ind'],
                               labels=np.unique(self.data.y), verbose=display_report,
                               plot=plot_roc,
                               save_status=save_file,
                               save_path=f'C:\\Users\\Anak\\PycharmProjects\\AdaptiveGraphStructureEmbedding\\Output\\Report\\{self.dataset}\\train_model\\',
                               file_name=report_test_file,
                               return_value_for_cv=return_report_stat_for_cv)

        return train_report, test_report


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
    torch.manual_seed(111)

    # dataset = 'cora'
    dataset = 'citeseer'

    # todo here>> convert torch geometric data to torch data
    my_new_model = MyNewModel(dataset, k_fold_split=args.k_fold_split , run_gcn_only=args.run_gcn_only)
    my_new_model.run_my_new_model()
