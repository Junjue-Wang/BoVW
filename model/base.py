# Date: 2019.07.15
# Author: Kingdrone
import numpy as np

class Base(object):
    def __init__(self, config):
        self.config = dict()
        self.set_default_config()
        self.update_config(config)

    def fit(self, x):
        """
        Args:
            x: an instance of numpy, 2D, [M, N]
        """
        raise NotImplementedError
    def predict(self, x):
        raise NotImplementedError

    def val(self, x, y):
        raise NotImplementedError

    def set_default_config(self):
        raise NotImplementedError

    def update_config(self, config):
        self.config.update(config)