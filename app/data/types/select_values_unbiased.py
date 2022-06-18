import logging
from typing import List
import numpy as np
import math
import random
from functools import partial

import torch
from torch import LongTensor
from torch.utils.data import DataLoader, TensorDataset

from app.data.data_type_abstract import DataTypeAbstract
from app.auxiliary_files.other_methods.util_functions import timeit

logger = logging.getLogger(__name__)


class DataTypeSelectValuesUnbiased(DataTypeAbstract):

    def __init__(self, config, *args):
        super().__init__(config, *args)
        self.shuffle = config['shuffle'] if 'shuffle' in config else True
        self.data_loader_args = config['data_loader'] if 'data_loader' in config else {}

    def create_data_loader(
            self,
            time_series_ids: List[str],
            time_series_lengths: List[int],
            is_train: bool
    ) -> DataLoader:
        # create data loader with the indexes of the selected possible values,
        # giving more importance to the ones before a pronounced spike
        # we do not create a new array to not waste memory (the transformation is done in collate_fn)
        # the indexes are in the way -> (time_series_index, time_position)
        indexes = []

        # load all time series in memory
        _, _, _, time_series_values_values_list, time_series_target_values_list = self.data_source.load_time_series_list(
            time_series_ids,
            times=False,
            init=False
        )

        if is_train:
            for time_series_id_index, time_series_id in enumerate(time_series_ids):
                time_series_values = time_series_values_values_list[time_series_id_index]
                time_series_target = time_series_target_values_list[time_series_id_index]
                max_time_series_cpu = np.max(time_series_values[0, :])
                for time_index in range(self.lag_size, time_series_values.shape[1] - self.prediction_size):
                    lag_values = time_series_values[0, (time_index - self.prediction_size):time_index]
                    prediction_values = time_series_target[0, time_index:(time_index + self.prediction_size)]

                    max_lag_values = np.max(lag_values)
                    max_prediction_values = np.max(prediction_values)

                    max_difference = np.abs(max_lag_values - max_prediction_values)
                    max_difference /= max_time_series_cpu

                    probability = 1 - math.exp(-max_difference) + 0.2

                    if random.random() < probability:
                        indexes.append((time_series_id_index, time_index))
        else:
            indexes = []
            for time_series_id_index, time_series_id in enumerate(time_series_ids):
                time_series_length = time_series_lengths[time_series_id_index]
                possible_time_values = [(time_series_id_index, time_position) for time_position in
                                        range(self.lag_size, time_series_length - self.prediction_size + 1)]
                indexes += possible_time_values

        logger.info(f'Total possible values {len(indexes)}')

        return DataLoader(
            dataset=TensorDataset(LongTensor(indexes)),
            shuffle=(is_train and self.shuffle),
            batch_size=self.batch_size,
            pin_memory=True,
            collate_fn=partial(
                self.collate_fn,
                time_series_ids=time_series_ids,
                time_series_values_values_list=time_series_values_values_list,
                time_series_target_values_list=time_series_target_values_list
            ),
            **self.data_loader_args
        )

    @timeit
    def collate_fn(
            self,
            samples,
            time_series_ids,
            time_series_values_values_list,
            time_series_target_values_list
    ):
        values = torch.zeros((len(samples), time_series_values_values_list[0].shape[0], self.lag_size))  # TODO does not work if list empty
        target = torch.zeros(len(samples), self.prediction_size)

        ids = []
        for index, sample in enumerate(samples):
            time_series_index, time_position = sample[0][0].item(), sample[0][1].item()
            ids.append((time_series_ids[time_series_index], time_position))
            values[index] = torch.from_numpy(time_series_values_values_list[time_series_index][:, (time_position - self.lag_size):time_position])
            target[index] = torch.from_numpy(time_series_target_values_list[time_series_index][0, time_position:(time_position + self.prediction_size)])

        return ids, values, target
