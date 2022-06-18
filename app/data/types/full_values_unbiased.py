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

logger = logging.getLogger(__name__)


class DataTypeFullValuesUnbiased(DataTypeAbstract):

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
        # create data loader with the indexes of all possible values
        # we do not create a new array to not waste memory (the transformation is done in collate_fn)
        # the indexes are in the way -> (time_series_index, time_position)
        maximum_per_time_series = min(time_series_lengths) - self.lag_size - self.prediction_size
        indexes = []
        for time_series_id_index in range(len(time_series_ids)):
            time_series_length = time_series_lengths[time_series_id_index]
            possible_time_values = [(time_series_id_index, time_position) for time_position in range(self.lag_size, time_series_length - self.prediction_size + 1)]
            if is_train:
                possible_time_values = random.sample(possible_time_values, maximum_per_time_series)
            indexes += possible_time_values
        logger.info(f'Total possible values {len(indexes)}')

        # load all time series in memory
        _, _, _, time_series_values_values_list, time_series_target_values_list = self.data_source.load_time_series_list(
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

    @timeit
    def get_final_predictions(self, all_predictions: np.ndarray):
        time_series_ids = []
        final_predictions = []

        last_time_series_id = None
        last_time_series_predictions = None
        for all_predictions_batch in all_predictions:
            batch_predictions_ids = all_predictions_batch[0]
            batch_predictions_values = all_predictions_batch[1]
            for prediction_index, (prediction_time_series_id, prediction_time_position) in enumerate(batch_predictions_ids):
                prediction_values = batch_predictions_values[prediction_index]
                if last_time_series_id is None:
                    last_time_series_id = prediction_time_series_id
                    last_time_series_predictions = [0] * self.lag_size + [prediction_values[0]]
                elif last_time_series_id == prediction_time_series_id:
                    last_time_series_predictions.append(prediction_values[0])
                else:
                    time_series_ids.append(last_time_series_id)
                    final_predictions.append(last_time_series_predictions)
                    last_time_series_id = prediction_time_series_id
                    last_time_series_predictions = [0] * self.lag_size + [prediction_values[0]]
        time_series_ids.append(last_time_series_id)
        final_predictions.append(last_time_series_predictions)

        del all_predictions

        # load time series
        _, _, time_series_initial_values_list, _, _ = self.data_source.load_time_series_list(time_series_ids)

        target_feature_index = self.data_source.get_target_feature_index()
        time_series_initial_values_list = [initial_values[target_feature_index] for initial_values in time_series_initial_values_list]

        return time_series_ids, time_series_initial_values_list, final_predictions
