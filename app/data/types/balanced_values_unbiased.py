import logging
from typing import List
from functools import partial
import random
from bisect import bisect
import numpy as np
import itertools

import torch
from torch import LongTensor
from torch.utils.data import DataLoader, TensorDataset

from app.data.data_type_abstract import DataTypeAbstract
from app.auxiliary_files.other_methods.util_functions import timeit

logger = logging.getLogger(__name__)


class DataTypeBalancedValuesUnbiased(DataTypeAbstract):

    def __init__(self, config, *args):
        super().__init__(config, *args)
        self.shuffle = config['shuffle'] if 'shuffle' in config else True
        self.data_loader_args = config['data_loader'] if 'data_loader' in config else {}
        self.window_size = 5

    def create_data_loader(
            self,
            time_series_ids: List[str],
            time_series_lengths: List[int],
            is_train: bool
    ) -> DataLoader:
        if is_train:
            target_feature_index = self.data_source.get_target_feature_index()
            indexes = []

            # load all time series in memory
            _, _, _, time_series_values_values_list, time_series_target_values_list = self.data_source.load_time_series_list(
                time_series_ids,
                times=False,
                init=False
            )

            for time_series_values_index in range(len(time_series_values_values_list)):
                time_series_values = time_series_values_values_list[time_series_values_index][target_feature_index]  # only target feature

                time_series_std = np.std(time_series_values)
                difference_intervals = [time_series_std * value for value in [0, 0.5, 2, 3]]
                difference_intervals_count = [0] * len(difference_intervals)
                difference_intervals_positions = [[] for _ in range(len(difference_intervals))]
                for time_position in range(max(self.window_size, self.lag_size), time_series_values.shape[0] - self.prediction_size + 1):
                    lag_mean = np.mean(time_series_values[(time_position - self.window_size):time_position])
                    difference = time_series_values[time_position] - lag_mean

                    interval_index = bisect(difference_intervals, difference)
                    if interval_index >= len(difference_intervals):
                        interval_index = len(difference_intervals) - 1
                    difference_intervals_count[interval_index] += 1
                    difference_intervals_positions[interval_index].append((time_series_values_index, time_position))

                total_samples = min(time_series_lengths) - self.lag_size - self.prediction_size
                total_samples //= 2

                samples_for_interval = total_samples // len(difference_intervals)
                for interval_index in range(len(difference_intervals)):
                    if len(difference_intervals_positions[interval_index]) == 0:
                        pass
                    elif len(difference_intervals_positions[interval_index]) < samples_for_interval:
                        difference_intervals_positions[interval_index] = [
                            random.choice(difference_intervals_positions[interval_index]) for _ in
                            range(samples_for_interval)]
                    else:
                        difference_intervals_positions[interval_index] = random.sample(
                            difference_intervals_positions[interval_index], samples_for_interval)

                all_points = list(itertools.chain.from_iterable(difference_intervals_positions))
                indexes += all_points
        else:
            indexes = []
            for time_series_id_index in range(len(time_series_ids)):
                time_series_length = time_series_lengths[time_series_id_index]
                possible_time_values = [(time_series_id_index, time_position) for time_position in
                                        range(self.lag_size, time_series_length - self.prediction_size + 1)]
                indexes += possible_time_values

            # load all time series in memory
            _, _, _, time_series_values_values_list, time_series_target_values_list = self.data_source.load_time_series_list(
                time_series_ids,
                times=False,
                init=False
            )

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
        target = torch.zeros(len(samples), self.prediction_size)  # TODO only works with one feature as target

        ids = []
        for index, sample in enumerate(samples):
            time_series_index, time_position = sample[0][0].item(), sample[0][1].item()
            ids.append((time_series_ids[time_series_index], time_position))
            values[index] = torch.from_numpy(time_series_values_values_list[time_series_index][:, (time_position - self.lag_size):time_position])
            target[index] = torch.from_numpy(time_series_target_values_list[time_series_index][time_position:(time_position + self.prediction_size)])

        return ids, values, target
