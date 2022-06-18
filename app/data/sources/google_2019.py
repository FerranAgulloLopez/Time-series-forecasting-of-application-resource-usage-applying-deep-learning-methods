import copy
import logging
import random
from os import listdir
from os.path import isfile, join
from typing import List

import numpy as np
from torch import LongTensor
from torch.utils.data import DataLoader, TensorDataset

from app.auxiliary_files.other_methods.util_functions import timeit
from app.auxiliary_files.data_methods.data_transformations import randomly_split_list_values_in_three
from app.auxiliary_files.data_methods.time_series_charts import perform_charts
from app.auxiliary_files.data_methods.time_series_preprocessing import perform_preprocessing_steps
from app.data.data_source_abstract import DataSourceAbstract

logger = logging.getLogger(__name__)


class DataSourceGoogle2019(DataSourceAbstract):

    def __init__(self, config: dict, device: str):
        super().__init__(config, device)
        self.directory_path = config['directory_path']
        self.batch_size = config['batch_size']
        self.target_feature = config['target_feature']
        time_feature = config['time_feature']

        # load header
        with open(join(self.directory_path, 'header.csv')) as file:
            lines = file.readlines()
            header = lines[0].replace('\n', '').split(',')

        # extract metadata and create features array
        self.time_feature_index = header.index(time_feature)
        del header[self.time_feature_index]
        self.features = header
        logger.info(f'Features: {self.features}')

        # split if required
        if 'split' in config:  # TODO do better
            self.split = config['split']
            self.global_preprocessing_steps = config['preprocessing_steps']['global']
            self.values_preprocessing_steps = config['preprocessing_steps']['values']
            self.target_preprocessing_steps = [{
                'name': 'select_feature',
                'feature': self.target_feature
            }] + config['preprocessing_steps']['target']

        # load all ids
        self.all_ids = [
            filename.replace('.npy', '') for filename in listdir(self.directory_path)
            if isfile(join(self.directory_path, filename))
            and filename.endswith('.npy')
        ]

        if 'subsample' in config:
            aux_all_ids = self.all_ids
            random.shuffle(aux_all_ids)
            aux_all_ids = aux_all_ids[:config['subsample']]
            self.all_ids = {_id: index for index, _id in enumerate(aux_all_ids)}
        else:
            self.all_ids = {_id: index for index, _id in enumerate(self.all_ids)}

        self.initialized = False

    # ---> Main methods

    def get_target_feature_index(self):
        return self.features.index(self.target_feature)

    def load(self) -> DataLoader:
        return self.__create_data_loader()

    def __load_all_time_series(self):
        self.all_time_series_times = []
        self.all_time_series_values_initial = []
        for time_series_id in self.all_ids.keys():
            time_series = np.load(f'{join(self.directory_path, str(time_series_id))}.npy')
            self.all_time_series_times.append(time_series[self.time_feature_index])
            self.all_time_series_values_initial.append(np.delete(time_series, [self.time_feature_index], axis=0))

        # global preprocess
        features, _, _, self.all_time_series_values_initial = perform_preprocessing_steps(
            # TODO do not delete time series
            self.global_preprocessing_steps,
            self.features,
            list(self.all_ids.keys()),
            self.all_time_series_times,
            self.all_time_series_values_initial
        )

        # create values
        self.all_time_series_values_values = copy.deepcopy(self.all_time_series_values_initial)
        self.features, _, _, self.all_time_series_values_values = perform_preprocessing_steps(
            # TODO do not delete time series
            self.values_preprocessing_steps,
            self.features,
            list(self.all_ids.keys()),
            self.all_time_series_times,
            self.all_time_series_values_values
        )

        # create target
        self.all_time_series_values_target = copy.deepcopy(self.all_time_series_values_initial)
        features, _, _, self.all_time_series_values_target = perform_preprocessing_steps(
            # TODO do not delete time series
            self.target_preprocessing_steps,
            self.features,
            list(self.all_ids.keys()),
            self.all_time_series_times,
            self.all_time_series_values_target
        )

    @timeit
    # load them in order of apparition inside the input list
    def load_time_series_list(self, time_series_ids_list: List[str], times=True, init=True):
        if not self.initialized:
            self.__load_all_time_series()
            self.initialized = True

        return time_series_ids_list, \
               [self.all_time_series_times[self.all_ids[time_series_id]] for time_series_id in time_series_ids_list], \
               [self.all_time_series_values_initial[self.all_ids[time_series_id]] for time_series_id in time_series_ids_list], \
               [self.all_time_series_values_values[self.all_ids[time_series_id]] for time_series_id in time_series_ids_list], \
               [self.all_time_series_values_target[self.all_ids[time_series_id]] for time_series_id in time_series_ids_list]

    def load_split(self) -> (List[str], List[str], List[str]):
        if not self.initialized:
            self.__load_all_time_series()
            self.initialized = True

        if self.split['name'] == 'specific':
            raise NotImplementedError()
        elif self.split['name'] == 'random':
            train_indexes, val_indexes, test_indexes = randomly_split_list_values_in_three(
                list(self.all_ids.keys()),
                self.config['split']['train_size'],
                self.config['split']['val_size']
            )
        else:
            raise Exception('Unknown type of split type ' + str(self.split['name']))

        # get lengths
        train_indexes_dict = {train_index: index for index, train_index in enumerate(train_indexes)}
        train_lengths = [None] * len(train_indexes)
        val_indexes_dict = {val_index: index for index, val_index in enumerate(val_indexes)}
        val_lengths = [None] * len(val_indexes)
        test_indexes_dict = {test_index: index for index, test_index in enumerate(test_indexes)}
        test_lengths = [None] * len(test_indexes)
        for time_series_id, time_series_index in self.all_ids.items():
            length = self.all_time_series_times[time_series_index].shape[0]
            if time_series_id in train_indexes_dict:
                train_lengths[train_indexes_dict[time_series_id]] = length
            elif time_series_id in val_indexes_dict:
                val_lengths[val_indexes_dict[time_series_id]] = length
            elif time_series_id in test_indexes_dict:
                test_lengths[test_indexes_dict[time_series_id]] = length
            else:
                raise Exception(f'{time_series_id} not found')

        return (train_indexes, train_lengths), (val_indexes, val_lengths), (test_indexes, test_lengths)

    def visualize(self, config: dict, output_path: str):
        data_loader = self.__create_data_loader()
        logger.debug(f'Total number of time series: {len(data_loader.dataset)}')
        perform_charts(config['visualization'], data_loader, self.get_features_labels(), output_path)

    def get_batch_size(self) -> int:
        return self.batch_size

    def get_features_labels(self) -> List[str]:
        return self.features

    def set_features_labels(self, features: List[str]) -> None:
        self.features = features

    # ---> Auxiliary methods

    def __create_data_loader(self) -> DataLoader:
        return DataLoader(
            dataset=TensorDataset(LongTensor(np.arange(len(list(self.all_ids.keys()))))),
            shuffle=True,
            batch_size=self.batch_size,
            pin_memory=True,
            collate_fn=self.collate_fn
        )

    def collate_fn(self, samples) -> (np.ndarray, np.ndarray, np.ndarray):
        time_series_ids_list = []
        time_series_times_list = []
        time_series_values_list = []
        for sample in samples:
            _id = list(self.all_ids.keys())[sample[0].item()]
            time_series = np.load(f'{join(self.directory_path, str(_id))}.npy')
            time_series_ids_list.append(_id)
            time_series_times_list.append(time_series[self.time_feature_index])
            time_series_values_list.append(np.delete(time_series, [self.time_feature_index], axis=0))
        return time_series_ids_list, time_series_times_list, time_series_values_list
