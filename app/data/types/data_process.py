import logging
from typing import List

import numpy as np
from torch.utils.data import DataLoader

from app.auxiliary_files.data_methods.time_series_preprocessing import perform_preprocessing_steps
from app.auxiliary_files.other_methods.util_functions import print_pretty_json
from app.data.data_source_abstract import DataSourceAbstract
from app.data.data_type_interface import DataTypeInterface

logger = logging.getLogger(__name__)


class DataTypeProcess(DataTypeInterface):

    def __init__(
            self,
            config: dict,
            data_source: DataSourceAbstract,
            output_path: str,
            device: str
    ):
        self.data_source = data_source
        self.config = config
        self.preprocessing_steps = config['preprocessing_steps']
        self.output_directory_path = config['output_directory_path']

    # ---> Main methods

    def load_data(self) -> None:
        # --> load time series
        data_loader = self.data_source.load()

        # --> preprocess time series
        total = 0
        count = 0
        for (time_series_ids_list, time_series_times_list, time_series_values_list) in data_loader:
            logger.info(f'Iteration {total}')
            total += len(time_series_ids_list)
            _, time_series_ids_list, time_series_times_list, time_series_values_list = perform_preprocessing_steps(
                self.preprocessing_steps,
                self.data_source.get_features_labels(),
                time_series_ids_list,
                time_series_times_list,
                time_series_values_list
            )
            # self.data_source.set_features_labels(features)

            # --> save preprocessed time series
            for time_series_index, time_series_id in enumerate(time_series_ids_list):
                times = time_series_times_list[time_series_index]
                values_without_time = time_series_values_list[time_series_index]
                values_with_time = np.concatenate((np.expand_dims(times, axis=0), values_without_time), axis=0)
                np.save(f'{self.output_directory_path}/{time_series_id}', values_with_time)

            count += len(time_series_ids_list)

        logger.info(f'Total processed time series: {total}; output ones: {count}')

    def show_info(self) -> None:
        print_pretty_json(self.config)
        self.data_source.show_info()

    def get_train_time_series_list(self) -> (List[str], List[np.ndarray], List[np.ndarray]):
        raise NotImplementedError()

    def get_val_time_series_list(self) -> (List[str], List[np.ndarray], List[np.ndarray]):
        raise NotImplementedError()

    def get_test_time_series_list(self) -> (List[str], List[np.ndarray], List[np.ndarray]):
        raise NotImplementedError()

    def get_train_data_loader(self, for_training: bool) -> DataLoader:
        raise NotImplementedError()

    def get_val_data_loader(self) -> DataLoader:
        raise NotImplementedError()

    def get_test_data_loader(self) -> DataLoader:
        raise NotImplementedError()

    def perform_train_output_visualization(self, all_predictions: np.ndarray) -> None:
        raise NotImplementedError()

    def perform_val_output_visualization(self, all_predictions: np.ndarray) -> None:
        raise NotImplementedError()

    def perform_test_output_visualization(self, all_predictions: np.ndarray) -> None:
        raise NotImplementedError()

    def get_number_samples(self) -> int:
        raise NotImplementedError()

    def get_number_features(self) -> int:
        raise NotImplementedError()

    def get_lag_size(self) -> int:
        raise NotImplementedError()

    def get_prediction_size(self) -> int:
        raise NotImplementedError()

    def visualize(self, output_path: str):
        raise NotImplementedError()