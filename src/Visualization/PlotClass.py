import matplotlib.pyplot as plt
import pickle
import os

class PlotClass:
    def __init__(self):
        '''
        usecase
        :plotting loss
        1.
            plot_class = PlotClass()
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
        self.hist = {}
        self.plt = plt

    def collect_hist(self, name, val):
        """

        :param name: str
        :param val: int, float
        :return:
        """
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
            if self.row_col[0] == 1 or self.row_col[1] == 1:
                ind = ax_tuple[0] if ax_tuple[0] != 0 else ax_tuple[1]
                self.axs[ind].set(xlabel='epochs',ylabel='val' ,title=name)
                self.axs[ind].plot(self.hist[name], label=name)
                self.axs[ind].legend()
            else:
                self.axs[ax_tuple[0]][ax_tuple[1]].set(xlabel='epochs',ylabel='val' ,title=name)
                self.axs[ax_tuple[0]][ax_tuple[1]].plot(self.hist[name], label=name)
                self.axs[ax_tuple[0]][ax_tuple[1]].legend()
        else:
            raise ValueError('use plot_hist instead of plot_each_hist')
        print()


    def show(self):
        """use with plot_each_hist"""
        self.plt.plot()

    def plot_hist(self, plot_title):
        self.axs.set(xlabel='epochs',ylabel='val' ,title=plot_title)
        for name, val_hist in self.hist.items():
            self.axs.plot(val_hist, label=name)
        self.plt.show()

    # def save_fig(self, path=r'Output/Plot/', name=None):
    def save_hist_with_pickel(self ,path=f'C:\\Users\\Anak\\PycharmProjects\\AdaptiveGraphStructureEmbedding\\Output\\Plot\\', name=None):
        assert name is not None, "name must be specified to avoid ambiguity"
        save_path = path+ name
        os.makedirs(path,exist_ok=True)
        with open(save_path, 'wb') as p:
            pickle.dump(self.hist, p)

    def save_fig(self, path=r'C:\Users\Anak\PycharmProjects\AdaptiveGraphStructureEmbedding\Output\Plot\\', name=None):
        # permission denied
        assert name is not None, "name must be specified to avoid ambiguity"
        save_path = path + name
        os.makedirs(path,exist_ok=True)
        self.fig.savefig(save_path, format= 'png')
