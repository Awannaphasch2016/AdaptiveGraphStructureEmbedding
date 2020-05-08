import matplotlib.pyplot as plt

def visualize_roc_curve(fpr, tpr, roc_auc, save_path=None, file_name=None, save_status=None):
    """
    refer back to the following link : https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html#sphx-glr-auto-examples-model-selection-plot-roc-py
    @param fpr: type = dict
    @param tpr: type = dict
    @param roc_auc: type = dict
    @return:
    """
    save_file = None
    if save_path is not None and isinstance(save_path, str):
        assert isinstance(file_name, str), "file_name must be specified to avoid ambiguity"
        save_file = save_path + f'roc_{file_name}.png'

    plt.figure()
    lw = 2
    colors = ['red', 'blue', 'black', 'green', 'yellow']
    for i in range(len(list(fpr.keys()))):
        plt.plot(fpr[i], tpr[i], color=colors[i],
                 lw=lw,
                 label=f'Class = {i}; ROC curve (area = {roc_auc[i][0]: .2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'{file_name}')
    plt.legend(loc="lower right")  # todo show legen of all class_roc_curve
    if save_status:
        print(f'saving roc to {save_file}..')
        plt.savefig(save_file)
    plt.show()
