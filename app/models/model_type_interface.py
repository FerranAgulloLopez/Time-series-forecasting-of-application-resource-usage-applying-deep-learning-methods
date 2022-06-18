import abc

import torch.nn as nn


# Interface for configs
class ModelTypeInterface(metaclass=abc.ABCMeta):

    @classmethod
    def __subclasshook__(cls, subclass):  # to check that the subclasses follow the interface
        return (hasattr(subclass, 'show_info') and
                callable(subclass.show_info) and
                hasattr(subclass, 'train_test') and
                callable(subclass.train_test) and
                hasattr(subclass, 'save_results') and
                callable(subclass.save_results) and
                hasattr(subclass, 'save_model') and
                callable(subclass.save_model) or
                NotImplemented)

    # ---> Main methods

    @abc.abstractmethod
    def __init__(self, config: dict, data_model, network: nn.Module, output_path: str, device: str):
        pass

    @abc.abstractmethod
    def show_info(self):
        raise NotImplementedError('Method not implemented in interface class')

    @abc.abstractmethod
    def train_test(self):
        raise NotImplementedError('Method not implemented in interface class')

    @abc.abstractmethod
    def save_results(self, visualize: bool):
        raise NotImplementedError('Method not implemented in interface class')

    @abc.abstractmethod
    def save_model(self):
        raise NotImplementedError('Method not implemented in interface class')
