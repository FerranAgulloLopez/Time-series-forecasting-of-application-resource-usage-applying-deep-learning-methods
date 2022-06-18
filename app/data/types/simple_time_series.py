import logging
from typing import List
from functools import partial
import random
import numpy as np

import torch
from torch import LongTensor
from torch.utils.data import DataLoader, TensorDataset

from app.data.data_type_abstract import DataTypeAbstract
from app.auxiliary_files.other_methods.util_functions import timeit
from app.auxiliary_files.other_methods.visualize import compare_multiple_lines_points_color_2

logger = logging.getLogger(__name__)


class DataTypeSimpleTimeSeries(DataTypeAbstract):

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
        indexes = list(range(len(time_series_ids)))
        logger.info(f'Total possible values {len(indexes)}')
        target_feature_index = self.data_source.get_target_feature_index()

        # load all time series in memory
        _, _, time_series_list, _, _ = self.data_source.load_time_series_list(
            time_series_ids,
            times=False,
            init=False
        )

        return DataLoader(
            dataset=TensorDataset(LongTensor(indexes)),
            shuffle=(is_train and self.shuffle),
            batch_size=self.batch_size,
            pin_memory=True,
            collate_fn=partial(
                self.collate_fn,
                time_series_ids=time_series_ids,
                time_series_list=[time_series[target_feature_index] for time_series in time_series_list]
            ),
            **self.data_loader_args
        )

    @timeit
    def collate_fn(
            self,
            samples,
            time_series_ids,
            time_series_list
    ):
        ids = []
        time_series_batch_list = []
        for index, sample in enumerate(samples):
            time_series_index = sample[0].item()
            ids.append(time_series_ids[time_series_index])
            time_series_batch_list.append(time_series_list[time_series_index])

        return ids, time_series_batch_list

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

        time_series_ids, time_series_predictions_list = all_predictions
        time_series_ids = time_series_ids[:number_samples]
        time_series_predictions_list = time_series_predictions_list[:number_samples]

        _, _, time_series_initial_values_list, _, _ = self.data_source.load_time_series_list(time_series_ids)

        # generate charts
        for time_series_index, time_series_id in enumerate(time_series_ids):
            time_series_values = time_series_initial_values_list[time_series_index][target_feature_index]
            time_series_predictions = time_series_predictions_list[time_series_index]

            # complete
            lines = [
                (
                    time_series_values,
                    np.arange(time_series_values.shape[0]),
                    self.output_values_color,
                    2,
                    'initial'
                ),
                (
                    time_series_predictions,
                    np.arange(time_series_predictions.shape[0]),
                    self.output_prediction_color_1,
                    2,
                    'prediction'
                )
            ]
            # """

            compare_multiple_lines_points_color_2(
                False,
                lines,
                [],
                'feature',
                'instant',
                f'{title} {time_series_id} complete',
                f'{self.output_path}/{title}_{time_series_id}_complete'
            )
            # """
            # partials
            partial_size = 500
            for init_position in range(0, time_series_values.shape[0], partial_size):
                real_size = time_series_values[init_position:(init_position + partial_size)].shape[0]
                lines = [
                    (
                        time_series_values[init_position:(init_position + partial_size)],
                        np.arange(init_position, init_position + real_size),
                        self.output_values_color,
                        2,
                        'initial'
                    ),
                    (
                        time_series_predictions[init_position:(init_position + partial_size)],
                        np.arange(init_position, init_position + real_size),
                        self.output_prediction_color_1,
                        2,
                        'prediction'
                    )

                ]
                # """
                # """
                compare_multiple_lines_points_color_2(
                    False,
                    lines,
                    [],
                    'feature',
                    'instant',
                    f'{title} {time_series_id} partial {init_position}-{init_position + real_size}',
                    f'{self.output_path}/{title}_{time_series_id}_partial_{init_position}-{init_position + real_size}'
                )

    @timeit
    def get_final_predictions(self, all_predictions: np.ndarray):
        pass
