import keras
from keras.callbacks import Callback

class Callback(keras.callbacks.Callback):
    """
    Callback base class
    """

    def __init__(self):
        """
        Initialzies callback
        """
        # Fill with callback params
        ...

    # Fill this functions with desired callback behaviour
    # params are number of epoch and dict of monitored values

    def on_epoch_end(self, epoch: int, logs: dict):
        ...

    def on_train_batch_end(self, batch: int, logs: dict):
        ...

    def on_test_batch_end(self, batch: int, logs: dict):
        ...

    def on_train_batch_begin(self, batch: int, logs: dict):
        ...

    def on_test_batch_begin(self, batch: int, logs: dict):
        ...
