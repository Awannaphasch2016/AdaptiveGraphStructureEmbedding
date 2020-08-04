import matplotlib.pyplot as plt
class PlotEmb():
    def __init__(self, plot_status, save_status):
        self.save_status = save_status
        self.plot_status = plot_status
        self.emb = {}

    def collect_emb(self, name, emb):
        if name not in self.emb:
            self.emb.setdefault(name, emb)
        else:
            raise ValueError('wrong ')

    def plot_all(self, save_path=None, title=None):
        assert isinstance(save_path, str)
        assert isinstance(title, str)
        file = save_path + title + '.png'

        for i, k in self.emb.items():
            # if i in ['test', 'train']:
            #     plt.scatter(k[:, 0], k[:, 1], label=i)
            if i == 'min_fake':
                plt.scatter(k[:, 0], k[:, 1], label=i, marker="o")
            elif i == 'min_real':
                plt.scatter(k[:, 0], k[:, 1], label=i, marker="x")
            elif i == 'maj':
                plt.scatter(k[:, 0], k[:, 1], label=i, marker="^")
            # else:
            #     raise ValueError('only accept 4 values')
        plt.legend()
        if self.save_status:
            plt.savefig(file)
        plt.title(title)
        plt.show()



# def plot_emb(x, y, title=None):
#     assert isinstance(x, dict), ''
#     assert isinstance(y, dict), ''
#     assert isinstance(title, str), ' '
#     plt.scatter(x['train'],y['train'], label='train')
#     plt.scatter(x['test'],y['test'], label='test')
#
#     plt.scatter(x['maj'],y['maj'], label='minority', marker="^")
#     plt.scatter(x['min'],y['min'], label='test', marker="^")
#
#     plt.title(title)
#     plt.show()
#
#     # axs[0].scatter(X[y == 0, 0], X[y == 0, 1], label="Class #0", alpha=0.5)
#     # axs[0].scatter(X[y == 1, 0], X[y == 1, 1], label="Class #1", alpha=0.5)
#     # axs[0].set_title('Original set')
#     # plot_decoration(axs[0])
#     #
#     #
