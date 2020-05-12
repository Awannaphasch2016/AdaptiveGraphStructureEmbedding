import itertools
import os
import time

import numpy as np
import pandas as pd
import torch

from arg_parser import args
from Model.bench_mark_model import BenchMark
from Model.gcn_gan_model import GCNGAN
import inspect
import sys

current_dir = os.path.dirname(
os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)


if __name__ == '__main__':

    np.random.seed(111)
    torch.manual_seed(111)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # main_epoch = [1]
    # gan_epoch = [1, 2]
    # preserved_edges_percent = [1.0, 0.75]
    # model_name_list = ['run_gcn', 'train_model']
    # dataset_list = ['cora']

    main_epoch = [100]
    gan_epoch = [1, 5, 10, 25, 50]
    preserved_edges_percent = [1.0, 0.75, 0.5, 0.25]
    model_name_list = ['run_gcn', 'train_mode']
    dataset_list = ['cora', 'citeseer']

    time_stamp = time.strftime("%Y%m%d-%H%M%S")

    epoch_pair = list(
        itertools.product(main_epoch, gan_epoch, preserved_edges_percent,
                          model_name_list, dataset_list))

    model_perameters_dict = {}
    experiment_df = None

    for i, (e, ge, percent, model_name, dataset) in enumerate(epoch_pair):
        main_epoch = e
        preserved_edges_percent = percent
        k_fold_split = 3
        gan_epoch = ge
        dataset = dataset

        if model_name == 'run_gcn':
            run_gcn = True
        elif model_name == 'train_model':
            run_gcn = False
            ge = 0
        else:
            raise ValueError('')

        print(f'parameters = ({dataset},{model_name},percent={percent},e={e},ge={ge})')

        dataset_dict = {
            'dataset': dataset,  # name of data
            'is_downsampled': args.is_downsampled,
        }
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
        if run_gcn:
            MyModelClass = BenchMark
            gan_epoch = None
            model_parameters_dict = {
                'model_name': model_name,
                'run_gcn': run_gcn,
                'main_epoch': main_epoch,
                'preserved_edges_percent': preserved_edges_percent,
                'time_stamp': time_stamp,
                'k_fold_split': k_fold_split,
                'device': device,
            }
        else:
            MyModelClass = GCNGAN
            model_parameters_dict = {
                'model_name': model_name,
                'run_gcn': run_gcn,
                'gan_epoch': gan_epoch,
                'main_epoch': main_epoch,
                'preserved_edges_percent': preserved_edges_percent,
                'time_stamp': time_stamp,
                'k_fold_split': k_fold_split,
                'device': device,
            }

        my_model_class = MyModelClass(dataset_dict, model_parameters_dict,
                                      boolean_dict)
        model_performance_summary = my_model_class.run_model()

        # model_performance_row_tuple_list.append(my_model_class.tuple_row_index)
        # model_performance_col_tuple = my_model_class.tuple_col_index
        if experiment_df is None:
            experiment_df = model_performance_summary
        else:
            experiment_df = pd.concat((experiment_df, model_performance_summary))

        print()

    if args.is_saved_experiment:
        file_path = os.path.dirname(
            os.getcwd()) + f'\\Output\\Report\\{time_stamp}_experiment.csv'

        experiment_df = experiment_df.sort_index(axis=0, level=[0, 1, 2, 3,4], ascending=False)

        print(f'save experiment to {file_path}')
        experiment_df.to_csv(file_path)


