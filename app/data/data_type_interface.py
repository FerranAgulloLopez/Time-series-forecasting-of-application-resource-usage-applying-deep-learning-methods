import abc

import numpy as np
from torch.utils.data import DataLoader

from app.data.data_source_abstract import DataSourceAbstract


# Interface for the data storage and behaviour
class DataTypeInterface(metaclass=abc.ABCMeta):

    @classmethod
    def __subclasshook__(cls, subclass):  # to check that the subclasses follow the interface
        return (hasattr(subclass, 'load_data') and
                callable(subclass.load_data) and
                hasattr(subclass, 'get_train_data_loader') and
                callable(subclass.get_train_data_loader) and
                hasattr(subclass, 'get_val_data_loader') and
                callable(subclass.get_val_data_loader) and
                hasattr(subclass, 'get_test_data_loader') and
                callable(subclass.get_test_data_loader) and
                hasattr(subclass, 'perform_train_output_visualization') and
                callable(subclass.perform_train_output_visualization) and
                hasattr(subclass, 'perform_val_output_visualization') and
                callable(subclass.perform_val_output_visualization) and
                hasattr(subclass, 'perform_test_output_visualization') and
                callable(subclass.perform_test_output_visualization) and
                hasattr(subclass, 'get_number_samples') and
                callable(subclass.get_number_samples) and
                hasattr(subclass, 'get_number_features') and
                callable(subclass.get_number_features) and
                hasattr(subclass, 'get_lag_size') and
                callable(subclass.get_lag_size) and
                hasattr(subclass, 'get_prediction_size') and
                callable(subclass.get_prediction_size) and
                hasattr(subclass, 'visualize') and
                callable(subclass.visualize) or
                NotImplemented)

    @abc.abstractmethod
    def __init__(
            self,
            config: dict,
            data_source: DataSourceAbstract,
            output_path: str,
            device: str
    ):
        pass

    # ---> Main methods

    @abc.abstractmethod
    def load_data(self) -> None:
        # load time series and do preprocessing
        # pre -> none
        # post -> returns nothing
        raise NotImplementedError('Method not implemented in interface class')

    # TODO redo documentation
    @abc.abstractmethod
    def show_info(self) -> None:
        raise NotImplementedError('Method not implemented in interface class')

    @abc.abstractmethod
    def get_train_data_loader(self, for_training: True) -> DataLoader:
        raise NotImplementedError('Method not implemented in interface class')

    @abc.abstractmethod
    def get_val_data_loader(self) -> DataLoader:
        raise NotImplementedError('Method not implemented in interface class')

    @abc.abstractmethod
    def get_test_data_loader(self) -> DataLoader:
        raise NotImplementedError('Method not implemented in interface class')

    @abc.abstractmethod
    def perform_train_output_visualization(self, all_predictions: np.ndarray) -> None:
        raise NotImplementedError('Method not implemented in interface class')

    @abc.abstractmethod
    def perform_val_output_visualization(self, all_predictions: np.ndarray) -> None:
        raise NotImplementedError('Method not implemented in interface class')

    @abc.abstractmethod
    def perform_test_output_visualization(self, all_predictions: np.ndarray) -> None:
        raise NotImplementedError('Method not implemented in interface class')

    @abc.abstractmethod
    def get_number_samples(self) -> int:
        raise NotImplementedError('Method not implemented in interface class')

    @abc.abstractmethod
    def get_number_features(self) -> int:
        raise NotImplementedError('Method not implemented in interface class')

    @abc.abstractmethod
    def get_lag_size(self) -> int:
        raise NotImplementedError('Method not implemented in interface class')

    @abc.abstractmethod
    def get_prediction_size(self) -> int:
        raise NotImplementedError('Method not implemented in interface class')

    @abc.abstractmethod
    def visualize(self, output_path: str):
        raise NotImplementedError('Method not implemented in interface class')
