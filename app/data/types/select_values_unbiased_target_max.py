import logging
from typing import List
import numpy as np
import math
from functools import partial
import random

import torch
from torch import LongTensor
from torch.utils.data import DataLoader, TensorDataset

from app.data.data_type_abstract import DataTypeAbstract
from app.auxiliary_files.other_methods.util_functions import timeit
from app.auxiliary_files.other_methods.visualize import compare_multiple_lines_points_color

logger = logging.getLogger(__name__)


class DataTypeSelectValuesUnbiasedTargetMax(DataTypeAbstract):

    def __init__(self, config, *args):
        super().__init__(config, *args)
        self.shuffle = config['shuffle'] if 'shuffle' in config else True
        self.data_loader_args = config['data_loader'] if 'data_loader' in config else {}

    def get_prediction_size(self) -> int:
        return 1

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
                    prediction_values = time_series_target[time_index:(time_index + self.prediction_size)]

                    max_lag_values = np.max(lag_values)
                    max_prediction_values = np.max(prediction_values)

                    max_difference = np.abs(max_lag_values - max_prediction_values)
                    max_difference /= max_time_series_cpu

                    probability = 1 - math.exp(-max_difference) + 0.2

                    if random.random() < probability:
                        indexes.append((time_series_id_index, time_index))
        else:
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
        target = torch.zeros(len(samples), 1)

        ids = []
        for index, sample in enumerate(samples):
            time_series_index, time_position = sample[0][0].item(), sample[0][1].item()
            ids.append((time_series_ids[time_series_index], time_position))
            values[index] = torch.from_numpy(time_series_values_values_list[time_series_index][:, (time_position - self.lag_size):time_position])
            target[index] = np.max(time_series_target_values_list[time_series_index][time_position:(time_position + self.prediction_size)])

        return ids, values, target

    def generate_prediction_charts(
            self,
            time_series_ids: List[str],
            time_series_lengths: List[int],
            all_predictions: np.ndarray,
            number_samples: int,
            title: str
    ):
        target_feature_index = self.data_source.get_target_feature_index()
        number_samples = number_samples if number_samples != -1 else len(time_series_ids)

        # get the time series to load
        to_load_ids = {}

        for all_predictions_batch in all_predictions:
            batch_predictions_ids = all_predictions_batch[0]
            batch_predictions_values = all_predictions_batch[1]
            for prediction_index, (prediction_time_series_id, prediction_time_position) in enumerate(batch_predictions_ids):
                prediction_values = batch_predictions_values[prediction_index]
                if prediction_time_series_id not in to_load_ids:
                    to_load_ids[prediction_time_series_id] = [(prediction_time_position, prediction_values)]
                else:
                    to_load_ids[prediction_time_series_id] += [(prediction_time_position, prediction_values)]

        # load time series
        time_series_ids_list, time_series_times_list, time_series_initial_values_list, time_series_values_values_list, time_series_target_values_list = self.data_source.load_time_series_list(list(to_load_ids.keys())[:number_samples])

        # generate charts
        for time_series_index, time_series_id in enumerate(time_series_ids_list):
            initial_values_time_series = time_series_initial_values_list[time_series_index]
            values_values_time_series = time_series_values_values_list[time_series_index]
            target_values_time_series = time_series_target_values_list[time_series_index]

            # complete
            lines = [
                (
                    initial_values_time_series[target_feature_index, :],
                    np.arange(initial_values_time_series.shape[1]),
                    self.output_initial_color,
                    1,
                    'initial'
                ),
                (
                    values_values_time_series[target_feature_index, :],
                    np.arange(values_values_time_series.shape[1]),
                    self.output_values_color,
                    1,
                    'values'
                ),
                (
                    target_values_time_series,
                    np.arange(target_values_time_series.shape[0]),
                    self.output_target_color,
                    2,
                    'target'
                )
            ]

            points = [(prediction_values[0], prediction_time_position) for (prediction_time_position, prediction_values)
                      in to_load_ids[time_series_id]]
            lines.append(
                (
                    [prediction_values[0] for (_, prediction_values) in to_load_ids[time_series_id]],
                    [prediction_time_position for (prediction_time_position, _) in to_load_ids[time_series_id]],
                    self.output_prediction_color_2,
                    1,
                    'prediction'
                )
            )

            compare_multiple_lines_points_color(
                False,
                lines,
                points,
                'feature',
                'instant',
                f'{title} {time_series_id} complete',
                f'{self.output_path}/{title}_{time_series_id}_complete'
            )

            # partials
            partial_size = 500
            for init_position in range(0, values_values_time_series.shape[1], partial_size):
                real_size = initial_values_time_series[0, init_position:(init_position + partial_size)].shape[0]
                lines = [
                    (
                        initial_values_time_series[target_feature_index, init_position:(init_position + partial_size)],
                        np.arange(init_position, init_position + real_size),
                        self.output_initial_color,
                        1,
                        'initial'
                    ),
                    (
                        values_values_time_series[target_feature_index, init_position:(init_position + partial_size)],
                        np.arange(init_position, init_position + real_size),
                        self.output_values_color,
                        1,
                        'values'
                    ),
                    (
                        target_values_time_series[init_position:(init_position + partial_size)],
                        np.arange(init_position, init_position + real_size),
                        self.output_target_color,
                        2,
                        'target'
                    )
                ]

                points = [(prediction_values[0], prediction_time_position) for
                          (prediction_time_position, prediction_values) in to_load_ids[time_series_id]
                          if init_position < prediction_time_position < (init_position + real_size)]
                lines.append(
                    (
                        [prediction_values[0] for (prediction_time_position, prediction_values) in
                         to_load_ids[time_series_id] if
                         init_position < prediction_time_position < (init_position + real_size)],
                        [prediction_time_position for (prediction_time_position, _) in to_load_ids[time_series_id] if
                         init_position < prediction_time_position < (init_position + real_size)],
                        self.output_prediction_color_2,
                        1,
                        'prediction'
                    )
                )

                compare_multiple_lines_points_color(
                    False,
                    lines,
                    points,
                    'feature',
                    'instant',
                    f'{title} {time_series_id} partial {init_position}-{init_position + real_size}',
                    f'{self.output_path}/{title}_{time_series_id}_partial_{init_position}-{init_position + real_size}'
                )
