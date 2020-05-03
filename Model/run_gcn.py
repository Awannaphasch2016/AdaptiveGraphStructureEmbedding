# import os,sys,inspect
# current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
# parent_dir = os.path.dirname(current_dir)
# sys.path.insert(0,parent_dir)
from src.Evaluation import report_performance
from src.Evaluation import get_total_roc_auc_score
import numpy as np

import src.Modeling.gcn as gcn_model
from src.Modeling.gcn import *
from src.Preprocessing import ModelInputData
from src.Visualization import PlotClass
import matplotlib.pyplot as plt
import time

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
    X_train_ind, X_test_ind, y_train, y_test = train_test_split(ind, y,
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


if __name__ == '__main__':
    np.random.seed(111)
    data, _ = torch.load(
        r'C:\Users\Anak\PycharmProjects\AdaptiveGraphStructureEmbedding\Notebook\data\Cora\Cora\processed\data.pt')

    data.y_before_relabel = data.y
    new_y = relabel_minority_and_majority_classes(data)
    data.y = new_y
    data.num_classes = np.unique(data.y).shape[0]

    model_input_data = ModelInputData(data)
    (trainning_selected_min_ind, trainning_selected_maj_ind), (
    test_selected_min_ind,
    test_selected_maj_ind) = model_input_data.prepare_ind_for_trainning_and_test_set()

    tmp = np.zeros_like(data.y)
    trainning_selected_min_ind = np.random.permutation(trainning_selected_min_ind)[:10]
    trainning_selected_maj_ind = np.random.permutation(trainning_selected_maj_ind)[:10]
    ind = np.concatenate((trainning_selected_min_ind, trainning_selected_maj_ind))
    tmp[ind] = 1
    tmp = np.random.permutation(tmp)
    data.train_mask  = torch.tensor(tmp).type(torch.BoolTensor)

    tmp = np.zeros_like(data.y)
    test_selected_min_ind = np.random.permutation(test_selected_min_ind)[:10]
    test_selected_maj_ind = np.random.permutation(test_selected_maj_ind)[:10]
    ind = np.concatenate((test_selected_min_ind, test_selected_maj_ind))
    tmp[ind] = 1
    tmp = np.random.permutation(tmp)
    data.test_mask  = torch.tensor(tmp).type(torch.BoolTensor)

    # =====================
    # ==for Gcn
    # ====================

    gcn = gcn_model.GCN(data)

    gcn.data.y = torch.tensor(gcn.data.y).type(torch.long)


    def test():
        gcn.model.eval()

        (emb_after_cov1, emb_after_cov2), accs = gcn.model(gcn.get_dgl_graph(),
                                                           gcn.data.x), []
        logits = F.log_softmax(emb_after_cov2, dim=1)

        for _, mask in gcn.data('train_mask', 'val_mask',
                                'test_mask'):  # torch_geometric
            pred = logits[mask].max(1)[1]
            acc = pred.eq(gcn.data.y[mask]).sum().item() / mask.sum().item()
            accs.append(acc)

        return accs

    def collect_performance(logits, minreal_minfake_majreal_y):

        y_pred_dict = {}
        y_score_dict = {}
        y_true_dict = {}

        accs_dict = {}
        aucs_dict = {}


        for name, mask in data('train_mask', 'test_mask'):
            pred = logits[mask].max(1)[1]
            y_true = minreal_minfake_majreal_y[mask]
            y_score = logits[mask].detach().numpy()


            y_pred_dict.setdefault(name, pred.detach().numpy())
            y_score_dict.setdefault(name, y_score)
            y_true_dict.setdefault(name, y_true.detach().numpy())

            acc = pred.eq(minreal_minfake_majreal_y[
                              mask]).sum().item() / mask.sum().item()
            if acc > 1:
                print()
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

    time_stamp = time.strftime("%Y%m%d-%H%M%S")
    plot_class = PlotClass()
    best_val_acc = test_acc = 0

    plot_class.set_subplots((3, 1))
    performance_history = {}
    for epoch in range(1, 201):

        # emb_after_conv1, emb_after_conv2 = self.gcn.train()
        gcn.model.train()
        emb_after_conv1 = gcn.model(gcn.get_dgl_graph(), data.x,
                                    get_conv1_emb=True)
        emb_after_conv2, logits = gcn.model(gcn.get_dgl_graph(),
                                            emb_after_conv1,
                                            run_discriminator=True)

        trainning_loss = gcn.loss_and_step(
            logits[data.train_mask],
            data.y[data.train_mask])

        # =====================
        # == gan test
        # =====================
        gcn.model.eval()
        (logits) = gcn.model(
            gcn.get_dgl_graph(), data.x,
            run_all=True)

        test_loss = gcn.loss(
            logits[data.test_mask],
            data.y[data.test_mask])

        # x_after_conv1, x_after_conv2 = gcn.model.train()
        # logits = gcn.discriminator(x_after_conv2)
        # gcn.loss_and_step(logits[data.train_mask],
        #                   data.y[data.train_mask])
        #
        # acc = test()
        # train_acc, val_acc, tmp_test_acc = test()
        #
        # if val_acc > best_val_acc:
        #     best_val_acc = val_acc
        #     test_acc = tmp_test_acc
        #
        # log = f'Epoch: {epoch:03d}, Train: {train_acc:.4f}, Val: {best_val_acc:.4f}, Test: {test_acc:.4f}'
        # print(log)


        accs_dict, aucs_dict, y_pred_dict, y_score_dict, y_true_dict = collect_performance(
            logits, data.y)

        print(f' Epoch: {epoch:03d},\n'
              f'Train: (acc={accs_dict["train_mask"]:.4f}, auc={aucs_dict["train_mask"]: .4f}, loss={trainning_loss:.4f}),\n'
              f'Test: ({accs_dict["test_mask"]:.4f}, auc={aucs_dict["test_mask"]:4f}, loss={test_loss:.4f})')

        plot_class.collect_hist('train_loss', trainning_loss)
        plot_class.collect_hist('test_loss', test_loss)
        plot_class.collect_hist('train_acc', accs_dict['train_mask'])
        plot_class.collect_hist('test_acc', accs_dict['test_mask'])
        plot_class.collect_hist('train_auc', aucs_dict['train_mask'])
        plot_class.collect_hist('test_auc', aucs_dict['test_mask'])


    plot_class.plot_each_hist((0, 0), name='train_loss')
    plot_class.plot_each_hist((0, 0), name='test_loss')
    plot_class.plot_each_hist((1, 0), name='train_acc')
    plot_class.plot_each_hist((1, 0), name='test_acc')
    plot_class.plot_each_hist((2, 0), name='train_auc')
    plot_class.plot_each_hist((2, 0), name='test_auc')
    plot_class.save_hist_with_pickel(
        name=f'from_run_gcn_key=train_loss_{time_stamp}.pickle', key='train_loss')
    plot_class.save_hist_with_pickel(
        name=f'from_run_gcn_key=test_loss_{time_stamp}.pickle', key='test_loss')
    plot_class.save_fig(name=f'from_run_gcn_{time_stamp}.png')
    plot_class.save_hist_with_pickel(
        name=f'from_run_gcn_key=train_acc_{time_stamp}.pickle', key='train_acc')
    plot_class.save_hist_with_pickel(
        name=f'from_run_gcn_key=test_acc_{time_stamp}.pickle', key='test_acc')
    plot_class.save_fig(name=f'from_run_gcn_{time_stamp}.png')
    plot_class.save_hist_with_pickel(
        name=f'from_run_gcn_key=train_auc_{time_stamp}.pickle', key='train_auc')
    plot_class.save_hist_with_pickel(
        name=f'from_run_gcn_key=test_auc_{time_stamp}.pickle', key='test_auc')
    plot_class.save_fig(name=f'from_run_gcn_{time_stamp}.png')

    print('=====train========')
    report_train_file = f'train_{time_stamp}'
    report_test_file = f'test_{time_stamp}'
    save_path = f'C:\\Users\\Anak\\PycharmProjects\\AdaptiveGraphStructureEmbedding\\Output\\Report\\run_gcn\\'
    report_performance(y_true_dict['train_mask'], y_pred_dict['train_mask'],
                       y_score_dict['train_mask'],
                       labels=np.unique(data.y), verbose=True,
                       plot=True,
                       save_path= save_path,
                       file_name=report_train_file)
    print('=====test======')
    report_performance(y_true_dict['test_mask'], y_pred_dict['test_mask'],
                       y_score_dict['test_mask'],
                       labels=np.unique(data.y), verbose=True,
                       plot=True,
                       save_path=save_path,
                       file_name=report_test_file)

