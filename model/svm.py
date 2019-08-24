# Date: 2019.07.22
# Author: Kingdrone
from model.base import Base
import numpy as np
from sklearn.svm import SVC
from model.metric import Accuracy, ConfMatrix, plot_cm
import logging
from model.dataloader import *

class SVM(Base):
    def __init__(self, config):
        super(SVM, self).__init__(config)
        self.model = SVC(**self.config['model'])
        self.num_classes = self.config['num_classes']
        self.mean = None
        self.std = None
        self.base_samples = None
        self.logger = logging.getLogger(self.__class__.__name__)

    def fit(self, x, y):
        self.base_samples = x.copy()
        # self.labels = y.copy()
        self.mean = np.mean(self.base_samples, axis=0)
        self.std = np.std(self.base_samples, axis=0)
        if self.config['normalize']:
            self.base_samples = (self.base_samples - self.mean) / self.std

        self.model.fit(self.base_samples, y)

    def _preprocess(self, x):
        return (x - self.mean) / self.std

    def predict(self, x):
        """
        Args:
            x: an instance of numpy, [N, M]

        Returns:
            res: an instance of numpy, [N, ], the cls res of KNN
        """
        self.logger.info("start predict")
        if self.config['normalize']:
            x = self._preprocess(x)
        y = self.model.predict(x)
        return y.reshape(-1)

    def val(self, x, y, show_cm=True):
        """

        Args:
            x: an instance of numpy, [N, M], validation input
            y: an instance of numpy, [N, 1], validation label

        Returns:
            acc: accuracy metric
        """
        self.logger.info('**Start Val!**')
        x_val = x.copy()
        y_val = y.copy()

        acc_metric = Accuracy(self.num_classes)
        cm_metric = ConfMatrix(self.num_classes)
        y_pred = self.predict(x_val)

        acc_metric.update(y_pred, y_val)
        cm_metric.update(y_pred, y_val)

        if show_cm:
            plot_cm(cm_metric.value())

        return dict(acc=acc_metric.value(), cm=cm_metric.value())

    def set_default_config(self):
        self.config.update(dict(
            num_classes=12,
            normalize=True,
            model=dict(
                C=2.0,
                cache_size=200,
                class_weight=None,
                coef0=0.0,
                decision_function_shape='ovo',
                degree=3,
                gamma='auto',
                kernel='rbf',
                max_iter=-1,
                probability=False,
                random_state=None,
                shrinking=True,
                tol=0.001,
                verbose=False
            )
        ))

if __name__ == '__main__':
    pg = SampleGenerator(root_dir=r'C:\Users\mi\Desktop\作业\Google dataset of SIRI-WHU_earth_im_tiff\12class_tif', kernel_size=32, stride=16,
                         extractors=[MeanSTD(), GLCM()])
    train_samples, train_labels, test_samples, test_labels = pg.dataset_generator([30, 30], restore=True)
    svm = SVM({})
    svm.fit(train_samples, train_labels)
    res = svm.val(test_samples, test_labels, False)
    print(res['acc'])
