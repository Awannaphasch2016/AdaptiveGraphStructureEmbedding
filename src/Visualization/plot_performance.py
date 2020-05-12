import matplotlib.pyplot as plt
from collections.abc import Iterable
import torch
import numbers
import numpy as np
import pickle
import os

class PlotPerformance:
    def __init__(self, is_plotted, is_saved_plot, **kwargs):
        '''
        usecase
        :plotting loss
        1orplot_class = PlotClass()
            plot_class.set_subplots()
            plot_class.collect_loss(loss_val)
            plot_class.plot_hist()
            plt.save_fig()
        2.
            plot_class = PlotClass()
            plot_class.set_subplots(3,2)
            plot_class.collect_loss(plot1_name, loss_val1)
            plot_class.collect_loss(plot2_name, loss_val2)
            plot_class.plot_each_hist(ax_tuple(1,2), name='something')
            plot_class.plot_each_hist(ax_tuple(3,1), name='something1')
            plot.show()
            plt.save_fig()

        '''
        assert isinstance(is_saved_plot, bool) , 'please specified save_plot attribute in PlotClass to be boolean'
        assert isinstance(is_plotted, bool ), f'is_plotted must have type = bool'

        self.hist = {}
        self.plt = plt
        self.is_saved_plot = is_saved_plot

    def collect_hist_using_list_of_name(self, dict_for_plot, ):

       assert isinstance(dict_for_plot, dict), ''

       for name, val in dict_for_plot.items():
           self.collect_hist(name, val)

    def plot_using_list_of_name(self, subplot_size,name_and_tuple_dict,save_file_name, title=None, save_path=None):
        assert isinstance(subplot_size, tuple), ''
        assert isinstance(name_and_tuple_dict, dict), ''
        assert isinstance(save_path , str), ''

        self.set_subplots(subplot_size)
        if title is not None:
            assert isinstance(title, str), ''
            self.fig.suptitle(title)
        for name, tuple_pos in name_and_tuple_dict.items():
            if isinstance(tuple_pos, Iterable):
                try:
                    for i in tuple_pos:
                        assert isinstance(i,
                                          tuple), 'tuple_pos canot be iterate over'
                        assert i[0] - 1 <= subplot_size[0], 'tuple_pos too largs'
                        assert i[1] - 1 <= subplot_size[1], 'tuple_pos too largs'
                        self.plot_each_hist(i, name=name)
                except:
                    assert isinstance(tuple_pos,
                                      tuple), 'tuple_pos canot be iterate over'
                    assert tuple_pos[0] - 1 <= subplot_size[
                        0], 'tuple_pos too largs'
                    assert tuple_pos[1] - 1 <= subplot_size[
                        1], 'tuple_pos too largs'
                    self.plot_each_hist(tuple_pos, name=name)
            else:
                    raise ValueError('tuple_pos canot be iterate over')
        self.save_hist_with_pickel(path=save_path, name=f'{save_file_name}.pickle')
        self.save_fig(
            name=f'{save_file_name}.png',
            path=save_path)
        self.plt.show()

    def collect_hist(self, name, val):
        """

        :param name: str
        :param val: int, float
        :return:
        """
        if not isinstance(val, numbers.Number):
            if isinstance(val,np.ndarray):
                val = val.flatten()
                assert val.ndim == 1, 'e'
                for i in val:
                    self.hist.setdefault(name, []).append(i)
            elif isinstance(val, torch.Tensor):
                try:
                    val = torch.flatten(val).detach().numpy()
                except:
                    val = torch.flatten(val)

                assert val.ndim == 1, 'e'
                for i in val:
                    self.hist.setdefault(name, []).append(i)

            else:
                raise ValueError('only accept val of type number or numpy or torch.Tensor')
        else:
            self.hist.setdefault(name, []).append(val)

    def set_subplots(self, row_col):
        self.row_col = row_col

        if self.row_col is None:
            self.fig, self.axs = plt.subplots()
        else:
            assert isinstance(self.row_col, tuple), "self.row_col must be tuple"
            self.fig, self.axs = plt.subplots(*self.row_col)
            # self.axs = [a for ax in self.axs for a in ax]

    def plot_each_hist(self, ax_tuple=None, name=None):
        assert isinstance(ax_tuple, tuple), "ax_ind must be tuple"
        assert name is not None, "name must be specified to avoid ambiguity"

        if self.row_col is not None:
            if self.row_col[0] == 1 and self.row_col[1] == 1:
                self.axs.set(ylabel='val' ,title=name)
                self.axs.plot(self.hist[name], label=name)
                self.axs.legend()
            elif self.row_col[0] == 1 or self.row_col[1] == 1:
                ind = ax_tuple[0] if ax_tuple[0] != 0 else ax_tuple[1]
                # self.axs[ind].set(xlabel='epochs',ylabel='val' ,title=name)
                self.axs[ind].set(ylabel='val', title=name)
                self.axs[ind].plot(self.hist[name], label=name)
                self.axs[ind].legend()
            else:
                # self.axs[ax_tuple[0]][ax_tuple[1]].set(xlabel='epochs',ylabel='val' ,title=name)
                self.axs[ax_tuple[0]][ax_tuple[1]].set(ylabel='val' ,title=name)
                self.axs[ax_tuple[0]][ax_tuple[1]].plot(np.array(self.hist[name]).flatten(), label=name)
                self.axs[ax_tuple[0]][ax_tuple[1]].legend()
        else:
            raise ValueError('use plot_hist instead of plot_each_hist')


    def show(self):
        """use with plot_each_hist"""
        self.plt.plot()

    def plot_hist(self, plot_title):
        self.axs.set(xlabel='epochs',ylabel='val' ,title=plot_title)
        for name, val_hist in self.hist.items():
            self.axs.plot(val_hist, label=name)
        self.plt.show()

    # def save_fig(self, path=r'Output/Plot/', name=None):
    def save_hist_with_pickel(self, name=None, key=None, path=None):
        '''name_of_file should reflect hist key that is dumped'''
        assert name is not None, "name must be specified to avoid ambiguity"
        assert path is not None, "save_path must be specified to avoid ambiguity"

        save_path = path + name
        if self.is_saved_plot:
            print(f'saving pickle to {save_path} .. ')
            os.makedirs(path,exist_ok=True)
            if key is not None:
                hist = self.hist[key]
            else:
                hist = self.hist
            with open(save_path, 'wb') as p:
                pickle.dump(hist, p)
        else:
            print(f'is_saved_plot is false => pickle is not saved to {save_path}')
    def save_fig(self, name=None, path=None):
        # permission denied
        assert name is not None, "name must be specified to avoid ambiguity"
        assert path is not None, "path must be specified to avoid ambiguity"
        save_path = path + name
        if self.is_saved_plot:
            print(f'saving figure to {save_path} .. ')
            os.makedirs(path,exist_ok=True)
            self.fig.savefig(save_path, format= 'png')
        else:
            print(f'is_saved_plot is false => figure is not saved to {save_path}')
