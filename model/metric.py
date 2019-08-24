from scipy import sparse
import numpy as np
import torch
import time
import matplotlib.pyplot as plt

labels = ['agriculture', 'commercial', 'harbor', 'idle_land', 'industrial', 'meadow', 'overpass', 'park', 'pond', 'residential', 'river', 'water']


def plot_cm(cm_arr):
    """
    plot confusion matrix
    :param cm_arr: an instance of 2D array,
    :return:
    """
    plt.figure(figsize=(12, 8), dpi=120)
    classes_num = len(cm_arr)
    tick_marks = np.array(range(classes_num)) + 0.5
    ind_array = np.arange(classes_num)
    x, y = np.meshgrid(ind_array, ind_array)

    for x_val, y_val in zip(x.flatten(), y.flatten()):
        c = cm_arr[y_val][x_val]
        if c > 0.01:
            plt.text(x_val, y_val, "%0.2f" % (c,), color='red', fontsize=12, va='center', ha='center')
    # offset the tick
    plt.gca().set_xticks(tick_marks, minor=True)
    plt.gca().set_yticks(tick_marks, minor=True)
    plt.gca().xaxis.set_ticks_position('none')
    plt.gca().yaxis.set_ticks_position('none')
    plt.grid(True, which='minor', linestyle='-')
    plt.gcf().subplots_adjust(bottom=0.15)


    plt.imshow(cm_arr, interpolation='nearest', cmap=plt.cm.binary)
    plt.title('Normalized confusion matrix')
    plt.colorbar()
    xlocations = np.array(range(classes_num))
    plt.xticks(xlocations, labels, rotation=90)
    plt.yticks(xlocations, labels)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    np.set_printoptions(precision=2)

    # show confusion matrix
    # plt.savefig('confusion_matrix.png', format='png')
    plt.show()

class Accuracy(object):
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.num_true = [0.] * self.num_classes
        self.num_total = [0.] * self.num_classes

    def update(self, y_pred, y_true):
        for i in range(self.num_classes):
            num_i_true = np.sum((y_pred == i) * (y_true == i))
            num_i_total = np.sum(y_true == i)
            self.num_true[i] += num_i_true
            self.num_total[i] += num_i_total

    def value(self):
        cls_acc = {}
        for i in range(self.num_classes):
            cls_acc[labels[i]] = float(self.num_true[i] / self.num_total[i])
        acc_mean = sum(self.num_true) / sum(self.num_total)
        return dict(mean=acc_mean, acc_i=cls_acc)

class ConfMatrix(object):
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self._total = sparse.coo_matrix((num_classes, num_classes), dtype=np.float32)

    def update(self, y_pred, y_true):
        """

        Args:
            y_pred: 1-D
            y_true: 1-D

        Returns:

        """
        v = np.ones_like(y_pred)
        cm = sparse.coo_matrix((v, (y_true, y_pred)), shape=(self.num_classes, self.num_classes), dtype=np.float32)
        self._total += cm

    def value(self):
        dense_cm = self._total.toarray()
        row_sum = np.sum(dense_cm, axis=1)
        dense_cm /= row_sum[:, None]
        return dense_cm
