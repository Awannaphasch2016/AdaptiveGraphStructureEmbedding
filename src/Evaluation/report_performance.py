import copy

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import auc
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve

from src.Evaluation import get_total_roc_auc_score
from src.Visualization.visualize_performance import visualize_roc_curve

class ReportPerformance:
    def __init__(self,
                 is_displayed_performance_table,
                 is_plotted_roc,
                 is_saved_performance_table,
                 is_saved_plotted_roc,
                 **kwargs,
                 ):
        self.check_plot_performance_argument()

        self.is_plotted_roc = is_plotted_roc
        self.is_displayed_performance_table = is_displayed_performance_table

        self.is_saved_plotted_roc = is_saved_plotted_roc
        self.is_saved_performance_table = is_saved_performance_table

    def report_performance(self,
                           y_true,
                           y_pred,
                           y_score,
                           labels,
                           return_value_for_cv=False,
                           save_path=None,
                           file_name=None):
        """
        usecase:
            report_performance( np.array([0,1,2]), np.array([0, 2,2]), np.array([0.4,0.3,0.3, 0.3,0.4,0.3, 0.3, 0.3,0.4]).reshape(3,-1) , np.array([0,1,2]) , is_displayed_performance_table=True)

        @param y_true: shape = (# of instance,); type = np.array or list
        @param y_pred:  shape = (# of instance,); type = np.array or list
        @param y_score: shape = (# of instance, # of class); type = np.array
        @param labels: shape = (# of labels) eg. [0, 1, 2]; we need to be implicit because sometimes some labels are not predicted.
        @param is_plotted_roc:
        @param return_value_for_cv:
        @return: desc = return value if return_value_for_cv is True;
               report_final_performance_report_np: type = numpy
               columns_of_performance_metric:  type = list: desc = list of performance metrics name
        """
        if isinstance(y_true, type(torch.tensor([1]))):
            y_true = y_true.cpu().detach().numpy()
        if isinstance(y_score, type(torch.tensor([1]))):
            y_score = y_score.cpu().detach().numpy()
        if isinstance(y_pred, type(torch.tensor([1]))):
            y_pred = y_pred.cpu().detach().numpy()

        assert isinstance(y_true, np.ndarray), ''
        assert isinstance(y_pred, np.ndarray), ''
        assert isinstance(y_score, np.ndarray), ''

        pd.set_option('max_columns', None)
        pd.set_option('expand_frame_repr', False)

        save_file = None
        if save_path is not None and isinstance(save_path, str):
            assert isinstance(file_name,
                              str), "file_name must be specified to avoid ambiguity"
            import os
            os.makedirs(save_path, exist_ok=True)
            save_file = save_path + f'{file_name}.csv'

        assert self.is_displayed_performance_table is not None, "self.is_displayed_performance_table must be specified to avoid ambiguity"

        assert len(labels) > 1, "minimum label = 2 (aka binary classification)"

        report_sklearn_classification_report = classification_report(y_true,
                                                                     y_pred,
                                                                     labels,
                                                                     output_dict=True)

        report_dict = copy.deepcopy(report_sklearn_classification_report)

        # # TODO wtf is this paragraph?
        del report_dict['accuracy']
        report_dict['accuracy'] = {
            'precision': report_sklearn_classification_report['accuracy'],
            'recall': None,
            'f1-score': None, 'support': None}

        # --------add micro avg
        micro_avg = {}
        for i, j in report_sklearn_classification_report.items():
            try:
                if float(i) in labels:
                    for x, y in j.items():
                        if x == 'precision':
                            micro_avg.setdefault('precision', []).append(
                                float(y))
                        elif x == 'f1-score':
                            micro_avg.setdefault('f1-score', []).append(
                                float(y))
                        elif x == 'recall':
                            micro_avg.setdefault('recall', []).append(float(y))
            except:
                pass

        divider = labels.shape[0]

        x = micro_avg.copy()
        for i, j in micro_avg.items():
            x[i] = sum(j) / divider
        micro_avg = x

        report_dict['micro average'] = {'precision': micro_avg['precision'],
                                        'recall': micro_avg['recall'],
                                        'f1-score': micro_avg['f1-score'],
                                        'support': None}

        report_df = pd.DataFrame(report_dict).round(2).transpose()

        # show support class and predicted classes
        ## note: np.unique output sorted value
        supported_class_np, supported_class_freq_np = np.unique(y_true,
                                                                return_counts=True)
        predicted_class_np, predicted_class_freq_np = np.unique(y_pred,
                                                                return_counts=True)

        support_class_df = pd.DataFrame(supported_class_freq_np,
                                        columns=['support'],
                                        index=supported_class_np)
        predicted_class_df = pd.DataFrame(predicted_class_freq_np,
                                          columns=['predicted'],
                                          index=predicted_class_np)

        report_support_pred_class = pd.concat(
            [support_class_df, predicted_class_df], axis=1)

        # show AUC
        ## normalized to probability: ( This is a hack; because roc_auc_score only accept probaility like y_score .

        # TODO figure out why roc score is very low? what did I do wrong?
        ## read how get_roc_curve() works
        ##  create per class roc_auc_score; output shape = [# of instances]
        fpr, tpr, roc_auc = self._get_roc_curve(pd.get_dummies(y_true).to_numpy(),
                                                y_score, np.unique(y_true).shape[0])

        roc_auc = {i: [j] for i, j in roc_auc.items()}

        # todo this is just avg not micro avg
        # TODO create macro_avg,
        # roc_auc['avg'] = np.array(
        #     [j for i in roc_auc.values() for j in i]).mean()

        roc_auc_df = pd.DataFrame.from_dict(roc_auc).transpose()
        roc_auc_df.columns = ['AUC']

        total_roc_auc_score = get_total_roc_auc_score(y_true, y_score)

        pred_each_class_ind = {i: np.where(y_pred == i)[0] for i in labels}
        acc_per_class_dict = {}

        for i, j in pred_each_class_ind.items():
            if j.shape[0] == 0:
                acc_per_class_dict[str(i)] = 0
            else:
                y_true_mask = y_true[j]
                y_pred_mask = y_pred[j]
                acc_per_class_dict[str(i)] = [
                    np.equal(y_true_mask, y_pred_mask).sum(0) /
                    y_pred_mask.shape[0]]

        acc_per_class_df = pd.DataFrame.from_dict(
            acc_per_class_dict).transpose()
        acc_per_class_df.columns = ['ACC']

        if self.is_plotted_roc:
            visualize_roc_curve(fpr,
                                tpr,
                                roc_auc,
                                save_path=save_path,
                                file_name=file_name,
                                save_status=self.is_saved_plotted_roc)


        report_and_auc = self._combine_report_and_auc(report_df,
                                                      report_support_pred_class,
                                                      roc_auc_df,
                                                      total_roc_auc_score,
                                                      acc_per_class_df,
                                                      return_value_for_cv,
                                                      save_file,
                                                      self.is_displayed_performance_table)

        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(y_true, y_pred)
        if self.is_displayed_performance_table:
            print(cm)

        if return_value_for_cv:
            return report_and_auc, cm

    def _combine_report_and_auc(self,
                                report_df,
                                report_support_pred_class,
                                roc_auc_df,
                                total_roc_auc_score,
                                acc_per_class_df,
                                return_value_for_cv,
                                save_file,
                                verbose):
        # create mask
        ## create mask for report_support_pred_class that have the smae index as report_df.index (fill with nan)
        support_pred_na_np = np.tile(np.nan, (
            report_df.shape[0], report_support_pred_class.shape[1]))
        acc_per_class_na_np = np.tile(np.nan, (
            report_df.shape[0], acc_per_class_df.shape[1]))

        tmp = np.array(list(report_df.index))
        acc_ind = np.where(tmp == 'accuracy')
        tmp[acc_ind] = 'acc/total'
        report_df.index = tmp

        report_support_pred_class_mask_with_nan_df = pd.DataFrame(
            support_pred_na_np,
            index=report_df.index,
            columns=report_support_pred_class.columns)
        report_support_pred_class_mask_with_nan_df.loc[
        :report_support_pred_class.shape[0],
        :] = report_support_pred_class.values
        report_support_pred_class_mask_with_nan_with_predicted_col_df = \
            report_support_pred_class_mask_with_nan_df[['predicted']]

        report_acc_per_class_mask_with_nan_df = pd.DataFrame(
            acc_per_class_na_np,
            index=report_df.index,
            columns=acc_per_class_df.columns)
        #  report_df.loc['acc/total']['precision'] return accuracy value => total_accuracy has only 1 column, and I just put it at precision columns
        report_acc_per_class_mask_with_nan_df.loc[:acc_per_class_df.shape[0],
        :] = acc_per_class_df.values
        report_acc_per_class_mask_with_nan_df.loc['acc/total']['ACC'] = \
            report_df.loc['acc/total']['precision']

        ## create maks for roc_auc_df to have same index as report_df.index (fill with nan)

        na_np = np.tile(np.nan, (
            report_df.shape[0], roc_auc_df.shape[1]))

        roc_auc_mask_with_nan_df = pd.DataFrame(na_np, index=report_df.index,
                                                columns=roc_auc_df.columns)

        roc_auc_mask_with_nan_df.loc[:roc_auc_df.shape[0],
        :] = roc_auc_df.values

        roc_auc_mask_with_nan_df.loc['acc/total'] = total_roc_auc_score
        roc_auc_mask_with_nan_df.loc['macro avg'] = float('nan')
        # print(na_df)

        merged_report_df = pd.concat([report_df,
                                      report_support_pred_class_mask_with_nan_with_predicted_col_df,
                                      report_acc_per_class_mask_with_nan_df,
                                      roc_auc_mask_with_nan_df], axis=1)
        # merged_report_df = report_df.merge(report_support_pred_class, how='outer', on=['support'], copy=False, right_index=True)
        if self.is_saved_performance_table:
            print(f"save report_performance to {save_file} ...")
            if save_file is not None:
                merged_report_df.to_csv(save_file)
        else:
            print(f"{save_file} is not save because is_saved_performance_table is False")

        if verbose:
            print(merged_report_df)

        if return_value_for_cv:
            return merged_report_df
            # return merged_report_df.to_numpy(), merged_report_df.columns, merged_report_df.index

    def _get_roc_curve(self,y_true, y_score, n_classes):
        """
        refer back to the following link : https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html#sphx-glr-auto-examples-model-selection-plot-roc-py

        @param y_true: type = numpy; desc = onehot vector
        @param y_score: type = numpy; desc = onehot vector
        @return:
        """

        # Compute ROC curve and ROC area for each clasu
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_true[:, i], y_score[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        return fpr, tpr, roc_auc

    def check_plot_performance_argument(self):
        pass
