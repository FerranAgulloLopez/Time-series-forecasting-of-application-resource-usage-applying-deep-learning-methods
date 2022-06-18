import logging
from typing import List

import numpy as np
from torch.utils.data import DataLoader

from app.auxiliary_files.other_methods.util_functions import print_pretty_json
from app.auxiliary_files.other_methods.visualize import compare_multiple_lines_points_color
from app.data.data_source_abstract import DataSourceAbstract
from app.data.data_type_interface import DataTypeInterface

logger = logging.getLogger(__name__)


class DataTypeAbstract(DataTypeInterface):

    def __init__(
            self,
            config: dict,
            data_source: DataSourceAbstract,
            output_path: str,
            device: str
    ):
        # --> save input params
        self.lag_size = config['lag_size']
        self.prediction_size = config['prediction_size']
        self.batch_size = config['batch_size']
        self.data_source = data_source
        self.config = config
        self.output_path = output_path
        self.device = device

        # output visualization
        self.output_train_samples = config['output_visualization']['train_samples']
        self.output_val_samples = config['output_visualization']['val_samples']
        self.output_test_samples = config['output_visualization']['test_samples']
        self.output_initial_color = config['output_visualization']['initial_color']
        self.output_values_color = config['output_visualization']['values_color']
        self.output_target_color = config['output_visualization']['target_color']
        self.output_prediction_color_1 = config['output_visualization']['prediction_color_1']
        self.output_prediction_color_2 = config['output_visualization']['prediction_color_2']

    # ---> Main methods

    def load_data(self) -> None:
        # --> load time series
        (self.train_ids, self.train_lengths), (self.val_ids, self.val_lengths), (self.test_ids, self.test_lengths) = self.data_source.load_split()

        # --> show some info
        logger.info(f'Number of training samples: {len(self.train_ids)}')
        logger.info(f'Number of validation samples:{len(self.val_ids)}')
        logger.info(f'Number of testing samples:{len(self.test_ids)}')

    def show_info(self) -> None:
        print_pretty_json(self.config)
        self.data_source.show_info()

    def get_train_data_loader(self, for_training: bool) -> DataLoader:
        return self.create_data_loader(
            self.train_ids,
            self.train_lengths,
            is_train=for_training
        )

    def get_val_data_loader(self) -> DataLoader:
        return self.create_data_loader(
            self.val_ids,
            self.val_lengths,
            is_train=False
        )

    def get_test_data_loader(self) -> DataLoader:
        return self.create_data_loader(
            self.test_ids,
            self.test_lengths,
            is_train=False
        )

    def perform_train_output_visualization(self, all_predictions: np.ndarray) -> None:
        self.generate_prediction_charts(
            self.train_ids,
            self.train_lengths,
            all_predictions,
            self.output_train_samples,
            'train_predictions'
        )

    def perform_val_output_visualization(self, all_predictions: np.ndarray) -> None:
        self.generate_prediction_charts(
            self.val_ids,
            self.val_lengths,
            all_predictions,
            self.output_val_samples,
            'val_predictions'
        )

    def perform_test_output_visualization(self, all_predictions: np.ndarray) -> None:
        self.generate_prediction_charts(
            self.test_ids,
            self.test_lengths,
            all_predictions,
            self.output_test_samples,
            'test_predictions'
        )

    def get_number_samples(self) -> int:
        return len(self.labels_time_series_dict)

    def get_number_features(self) -> int:
        return self.data_source.get_number_features()

    def get_lag_size(self) -> int:
        return self.lag_size

    def get_prediction_size(self) -> int:
        return self.prediction_size

    def visualize(self, output_path: str):
        raise NotImplementedError()

    # ---> Auxiliary methods

    def create_data_loader(self,
                           time_series_indexes: List[str],
                           time_series_lengths: List[int],
                           is_train: bool
                           ) -> DataLoader:
        raise Exception('Not implemented in abstract class')

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
        del all_predictions

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
            # """
            points = []
            if next(iter(to_load_ids[time_series_id]))[1].shape[0] > 1:
                for (prediction_time_position, prediction_values) in to_load_ids[time_series_id]:
                    lines.append(
                        (
                            prediction_values,
                            np.arange(prediction_time_position, prediction_time_position + prediction_values.shape[0]),
                            self.output_prediction_color_1,
                            2,
                            None
                        )
                    )
            else:
                points = [(prediction_values[0], prediction_time_position) for
                          (prediction_time_position, prediction_values) in to_load_ids[time_series_id]]
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
                # """
                points = []
                if next(iter(to_load_ids[time_series_id]))[1].shape[0] > 1:
                    for (prediction_time_position, prediction_values) in to_load_ids[time_series_id]:
                        if init_position < prediction_time_position < (init_position + real_size):
                            if prediction_values.shape[0] > 1:
                                lines.append(
                                    (
                                        prediction_values,
                                        np.arange(prediction_time_position, prediction_time_position + prediction_values.shape[0]),
                                        self.output_prediction_color_1,
                                        2,
                                        None
                                    )
                                )
                else:
                    points = [(prediction_values[0], prediction_time_position) for
                              (prediction_time_position, prediction_values) in to_load_ids[time_series_id]
                              if init_position < prediction_time_position < (init_position + real_size)]
                    lines.append(
                        (
                            [prediction_values[0] for (prediction_time_position, prediction_values) in
                             to_load_ids[time_series_id] if
                             init_position < prediction_time_position < (init_position + real_size)],
                            [prediction_time_position for (prediction_time_position, _) in to_load_ids[time_series_id]
                             if init_position < prediction_time_position < (init_position + real_size)],
                            self.output_prediction_color_2,
                            1,
                            'prediction'
                        )
                    )
                # """
                compare_multiple_lines_points_color(
                    False,
                    lines,
                    points,
                    'feature',
                    'instant',
                    f'{title} {time_series_id} partial {init_position}-{init_position + real_size}',
                    f'{self.output_path}/{title}_{time_series_id}_partial_{init_position}-{init_position + real_size}'
                )

    def collate_fn(self, *args):
        raise Exception('Not implemented in abstract class')
