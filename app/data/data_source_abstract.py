import abc
from typing import List

from torch.utils.data import DataLoader

from app.auxiliary_files.other_methods.util_functions import print_pretty_json


# Interface for loading data from distinct types of sources
class DataSourceAbstract(metaclass=abc.ABCMeta):

    @classmethod
    def __subclasshook__(cls, subclass):  # to check that the subclasses follow the interface  // TODO update this
        return (hasattr(subclass, 'load') and
                callable(subclass.load) and
                hasattr(subclass, 'visualize') and
                callable(subclass.visualize) and
                hasattr(subclass, 'get_features_labels') and
                callable(subclass.get_features_labels) or
                NotImplemented)

    @abc.abstractmethod
    def __init__(self, config: dict, device: str):
        self.config = config

    # Main methods

    @abc.abstractmethod
    def load(self) -> DataLoader:
        # load time series and filter features
        # pre -> none
        # post -> returns two lists,  // TODO update this
        #         - numpy arrays with the values of the time series
        #         - strings with the identifiers of the time series
        raise NotImplementedError('Method not implemented in interface class')

    def show_info(self) -> None:
        print_pretty_json(self.config)

    @abc.abstractmethod
    def visualize(self, config: dict, output_path: str) -> None:
        # specific visualization of the dataset
        # pre -> none
        # post -> creates the desired charts
        raise NotImplementedError('Method not implemented in interface class')

    def get_number_features(self) -> int:
        return len(self.get_features_labels())

    @abc.abstractmethod
    def get_batch_size(self) -> int:
        raise NotImplementedError('Method not implemented in interface class')

    @abc.abstractmethod
    def get_features_labels(self) -> List[str]:
        # return the features labels
        # pre -> load method has been called
        # post -> returns the features labels
        raise NotImplementedError('Method not implemented in interface class')

    @abc.abstractmethod
    def set_features_labels(self, features: List[str]) -> None:
        # TODO update documentation
        raise NotImplementedError('Method not implemented in interface class')
