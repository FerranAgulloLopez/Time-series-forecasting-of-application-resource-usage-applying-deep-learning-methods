import logging
from typing import List
from functools import partial
import random
import numpy as np
from bisect import bisect
import pandas as pd
from datetime import datetime

import torch
from torch import LongTensor
from torch.utils.data import DataLoader, TensorDataset

from app.data.data_type_abstract import DataTypeAbstract
from app.auxiliary_files.other_methods.visualize import compare_multiple_lines_color, compare_multiple_lines_rectangles_color
from app.auxiliary_files.other_methods.util_functions import timeit
from app.auxiliary_files.data_methods.time_encodings import time_features

logger = logging.getLogger(__name__)


class DataTypeFullValuesUnbiasedIntervalsTransformer(DataTypeAbstract):

    def __init__(self, config, *args):
        super().__init__(config, *args)
        self.shuffle = config['shuffle'] if 'shuffle' in config else True
        self.data_loader_args = config['data_loader'] if 'data_loader' in config else {}
        self.intervals = config['intervals']
        self.decoder_input_lag_size = self.lag_size // 2
        self.prediction_size = 3  # transformer real prediction
        self.time_reduction = config['time_reduction'] if 'time_reduction' in config else 1

        self.log_transformation = config['log_transformation'] if 'log_transformation' in config else False

        if self.log_transformation:
            self.intervals = list(np.log(np.asarray(self.intervals) + 0.00000000001))

    def get_prediction_size(self) -> int:
        return len(self.intervals)

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
            possible_time_values = [(time_series_id_index, time_position) for time_position in
                                    range(self.lag_size, time_series_length - self.prediction_size + 1)]
            if is_train:
                possible_time_values = random.sample(possible_time_values, maximum_per_time_series)
            indexes += possible_time_values

        # load all time series in memory
        _, time_series_times_list, _, time_series_values_values_list, time_series_target_values_list = self.data_source.load_time_series_list(
            time_series_ids,
            times=False,
            init=False
        )

        # encode time
        for time_series_times_index, time_series_times in enumerate(time_series_times_list):
            time_series_times = time_series_times // self.time_reduction
            time_series_times = [datetime.fromtimestamp(time_series_times[timestamp_index]) for timestamp_index in range(time_series_times.shape[0])]
            time_series_times = pd.DataFrame({'date': time_series_times})
            time_series_times_list[time_series_times_index] = time_features(time_series_times, timeenc=1, freq='T')

        logger.info(f'Total possible values {len(indexes)}')

        return DataLoader(
            dataset=TensorDataset(LongTensor(indexes)),
            shuffle=(is_train and self.shuffle),
            batch_size=self.batch_size,
            pin_memory=True,
            collate_fn=partial(
                self.collate_fn_train if is_train else self.collate_fn_pred,
                time_series_ids=time_series_ids,
                time_series_times_list=time_series_times_list,
                time_series_values_values_list=time_series_values_values_list,
                time_series_target_values_list=time_series_target_values_list
            ),
            **self.data_loader_args
        )

    @timeit
    def collate_fn_train(
            self,
            samples,
            time_series_ids,
            time_series_times_list,
            time_series_values_values_list,
            time_series_target_values_list
    ):
        number_features = time_series_values_values_list[0].shape[0]  # TODO does not work if list empty
        number_time_features = 5
        values_x = torch.zeros((len(samples), self.lag_size, number_features))
        values_x_mark = torch.zeros((len(samples), self.lag_size, number_time_features))
        values_y = torch.zeros((len(samples), self.decoder_input_lag_size + self.prediction_size, number_features))
        values_y_mark = torch.zeros((len(samples), self.decoder_input_lag_size + self.prediction_size, number_time_features))
        target = torch.zeros(len(samples), dtype=torch.long)

        ids = []
        for index, sample in enumerate(samples):
            time_series_index, time_position = sample[0][0].item(), sample[0][1].item()
            ids.append((time_series_ids[time_series_index], time_position))

            values_x[index] = torch.from_numpy(np.transpose(time_series_values_values_list[time_series_index][:, (time_position - self.lag_size):time_position]))
            values_x_mark[index] = torch.from_numpy(time_series_times_list[time_series_index][(time_position - self.lag_size):time_position])

            values_y[index] = torch.from_numpy(np.transpose(time_series_values_values_list[time_series_index][:, (time_position - self.decoder_input_lag_size):(time_position + self.prediction_size)]))
            values_y_mark[index] = torch.from_numpy(time_series_times_list[time_series_index][(time_position - self.decoder_input_lag_size):(time_position + self.prediction_size)])

            max_target_value = np.max(time_series_target_values_list[time_series_index][time_position:(time_position + self.prediction_size)])
            interval_index = bisect(self.intervals, max_target_value)
            target[index] = interval_index

        return ids, values_x, values_x_mark, values_y, values_y_mark, target

    @timeit
    def collate_fn_pred(
            self,
            samples,
            time_series_ids,
            time_series_times_list,
            time_series_values_values_list,
            time_series_target_values_list
    ):
        number_features = time_series_values_values_list[0].shape[0]  # TODO does not work if list empty
        number_time_features = 5
        values_x = torch.zeros((len(samples), self.lag_size, number_features))
        values_x_mark = torch.zeros((len(samples), self.lag_size, number_time_features))
        values_y = torch.zeros((len(samples), self.decoder_input_lag_size, number_features))
        values_y_mark = torch.zeros((len(samples), self.decoder_input_lag_size + self.prediction_size, number_time_features))
        target = torch.zeros(len(samples), dtype=torch.long)

        ids = []
        for index, sample in enumerate(samples):
            time_series_index, time_position = sample[0][0].item(), sample[0][1].item()
            ids.append((time_series_ids[time_series_index], time_position))

            values_x[index] = torch.from_numpy(np.transpose(time_series_values_values_list[time_series_index][:, (time_position - self.lag_size):time_position]))
            values_x_mark[index] = torch.from_numpy(time_series_times_list[time_series_index][(time_position - self.lag_size):time_position])

            values_y[index] = torch.from_numpy(np.transpose(time_series_values_values_list[time_series_index][:, (time_position - self.decoder_input_lag_size):time_position]))
            values_y_mark[index] = torch.from_numpy(time_series_times_list[time_series_index][(time_position - self.decoder_input_lag_size):(time_position + self.prediction_size)])

            max_target_value = np.max(
                time_series_target_values_list[time_series_index][time_position:(time_position + self.prediction_size)])
            interval_index = bisect(self.intervals, max_target_value)
            target[index] = interval_index

        return ids, values_x, values_x_mark, values_y, values_y_mark, target


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
            for prediction_index, (prediction_time_series_id, prediction_time_position) in enumerate(
                    batch_predictions_ids):
                prediction_values = batch_predictions_values[prediction_index]
                if prediction_time_series_id not in to_load_ids:
                    to_load_ids[prediction_time_series_id] = [(prediction_time_position, prediction_values)]
                else:
                    to_load_ids[prediction_time_series_id] += [(prediction_time_position, prediction_values)]

        # load time series
        time_series_ids_list, time_series_times_list, time_series_initial_values_list, time_series_values_values_list, time_series_target_values_list = self.data_source.load_time_series_list(
            list(to_load_ids.keys())[:number_samples])

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
                    3,
                    'raw'
                ),
                (
                    values_values_time_series[target_feature_index, :],
                    np.arange(values_values_time_series.shape[1]),
                    self.output_values_color,
                    3,
                    'values series'
                ),
                (
                    target_values_time_series,
                    np.arange(target_values_time_series.shape[0]),
                    self.output_target_color,
                    3,
                    'target series'
                )
            ]
            # """
            rectangles = []
            init_position_interval = next(iter(to_load_ids[time_series_id]))[0]
            current_interval = np.argmax(next(iter(to_load_ids[time_series_id]))[1])
            for (prediction_time_position, prediction_values) in to_load_ids[time_series_id]:
                interval_index = np.argmax(prediction_values)
                if interval_index != current_interval:
                    height = self.intervals[interval_index + 1] - self.intervals[interval_index] \
                        if (interval_index + 1) < len(self.intervals) \
                        else self.intervals[interval_index] - self.intervals[interval_index - 1]
                    rectangles.append(
                        (
                            init_position_interval,  # x
                            self.intervals[interval_index],  # y
                            prediction_time_position - init_position_interval,  # width
                            height,  # height
                            0,  # angle
                            {'facecolor': (1, 0, 0, 0.2)}  # args
                        )
                    )
                    init_position_interval = prediction_time_position
                    current_interval = interval_index
            height = self.intervals[interval_index + 1] - self.intervals[interval_index] \
                if (interval_index + 1) < len(self.intervals) \
                else self.intervals[interval_index] - self.intervals[interval_index - 1]
            rectangles.append(
                (
                    init_position_interval,  # x
                    self.intervals[interval_index],  # y
                    prediction_time_position - init_position_interval,  # width
                    height,  # height
                    0,  # angle
                    {'facecolor': (1, 0, 0, 0.2)}  # args
                )
            )

            compare_multiple_lines_rectangles_color(
                False,
                lines,
                rectangles,
                'y',
                'time',
                f'',
                f'{self.output_path}/{title}_{time_series_id}_complete'
            )
            # """
            # partials
            partial_size = 500
            for init_position in range(0, values_values_time_series.shape[1], partial_size):
                real_size = initial_values_time_series[0, init_position:(init_position + partial_size)].shape[0]
                lines = [
                    (
                        initial_values_time_series[target_feature_index, init_position:(init_position + partial_size)],
                        np.arange(init_position, init_position + real_size),
                        self.output_initial_color,
                        3,
                        'raw'
                    ),
                    (
                        values_values_time_series[target_feature_index, init_position:(init_position + partial_size)],
                        np.arange(init_position, init_position + real_size),
                        self.output_values_color,
                        3,
                        'values series'
                    ),
                    (
                        target_values_time_series[init_position:(init_position + partial_size)],
                        np.arange(init_position, init_position + real_size),
                        self.output_target_color,
                        3,
                        'target series'
                    )
                ]
                # """
                rectangles = []
                init_position_interval = None
                current_interval = None
                first_out = True
                for (prediction_time_position, prediction_values) in to_load_ids[time_series_id]:
                    if init_position < prediction_time_position < (init_position + real_size):
                        interval_index = np.argmax(prediction_values)
                        if init_position_interval is None:
                            init_position_interval = prediction_time_position
                            current_interval = interval_index
                        elif interval_index != current_interval:
                            height = self.intervals[interval_index + 1] - self.intervals[interval_index] \
                                if (interval_index + 1) < len(self.intervals) \
                                else self.intervals[interval_index] - self.intervals[interval_index - 1]
                            rectangles.append(
                                (
                                    init_position_interval,  # x
                                    self.intervals[interval_index],  # y
                                    prediction_time_position - init_position_interval,  # width
                                    height,  # height
                                    0,  # angle
                                    {'facecolor': (1, 0, 0, 0.2)}  # args
                                )
                            )
                            init_position_interval = prediction_time_position
                            current_interval = interval_index
                    elif first_out and prediction_time_position > (init_position + real_size) and init_position_interval is not None:
                        first_out = False
                        height = self.intervals[interval_index + 1] - self.intervals[interval_index] \
                            if (interval_index + 1) < len(self.intervals) \
                            else self.intervals[interval_index] - self.intervals[interval_index - 1]
                        rectangles.append(
                            (
                                init_position_interval,  # x
                                self.intervals[interval_index],  # y
                                prediction_time_position - init_position_interval,  # width
                                height,  # height
                                0,  # angle
                                {'facecolor': (1, 0, 0, 0.2)}  # args
                            )
                        )
                        init_position_interval = prediction_time_position
                        current_interval = interval_index
                # """
                compare_multiple_lines_rectangles_color(
                    False,
                    lines,
                    rectangles,
                    'y',
                    'time',
                    f'',
                    f'{self.output_path}/{title}_{time_series_id}_partial_{init_position}-{init_position + real_size}'
                )

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
                interval_index = np.argmax(prediction_values)
                interval_value = self.intervals[interval_index]
                if last_time_series_id is None:
                    last_time_series_id = prediction_time_series_id
                    last_time_series_predictions = [0] * self.lag_size + [interval_value]
                elif last_time_series_id == prediction_time_series_id:
                    last_time_series_predictions.append(interval_value)
                else:
                    time_series_ids.append(last_time_series_id)
                    final_predictions.append(last_time_series_predictions)
                    last_time_series_id = prediction_time_series_id
                    last_time_series_predictions = [0] * self.lag_size + [interval_value]
        time_series_ids.append(last_time_series_id)
        final_predictions.append(last_time_series_predictions)

        del all_predictions

        # load time series
        _, _, time_series_initial_values_list, _, _ = self.data_source.load_time_series_list(time_series_ids)

        target_feature_index = self.data_source.get_target_feature_index()
        time_series_initial_values_list = [initial_values[target_feature_index] for initial_values in time_series_initial_values_list]

        if self.log_transformation:
            for time_series_index in range(len(time_series_initial_values_list)):
                final_predictions[time_series_index] = np.exp(np.asarray(final_predictions[time_series_index])) - 0.00000000001
                for time_position in range(1, final_predictions[time_series_index].shape[0]):
                    if final_predictions[time_series_index][time_position] == np.Infinity:
                        final_predictions[time_series_index][time_position] = final_predictions[time_series_index][time_position - 1]

        return time_series_ids, time_series_initial_values_list, final_predictions