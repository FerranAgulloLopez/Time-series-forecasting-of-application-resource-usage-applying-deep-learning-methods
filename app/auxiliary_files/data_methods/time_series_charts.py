import logging
import random
from typing import List

import numpy as np
from scipy.stats import spearmanr
from torch.utils.data import DataLoader

from app.auxiliary_files.other_methods.util_functions import \
    print_pretty_json, \
    save_json
from app.auxiliary_files.other_methods.visualize import \
    plot_hist, \
    compare_multiple_lines_matrix, \
    compare_multiple_lines, \
    plot_correlation_matrix, \
    plot_bar
from app.auxiliary_files.data_methods.data_transformations import MeanStdIterativeComputation

logger = logging.getLogger(__name__)


def perform_charts(config: dict, data_loader: DataLoader, features: List[str], output_path: str):
    charts = []
    for chart_config in config:
        name = chart_config['name']
        charts.append(eval(name)(chart_config, features, output_path))

    count = 0
    for (time_series_ids_list, time_series_times_list, time_series_values_list) in data_loader:
        logger.info(f'Iteration {count}')
        for chart in charts:
            chart.process_data(
                time_series_ids_list,
                time_series_times_list,
                time_series_values_list
            )
        count += len(time_series_ids_list)

    for chart in charts:
        logger.info(f'Showing {chart.__class__.__name__} results')
        chart.visualize()


class ChartGeneralLook:
    def __init__(self, config: dict, features: List[str], output_path):
        self.total_number = config['number']
        self.group_size = config['group_size']
        self.saved_time_series_ids_list = []
        self.saved_time_series_times_list = []
        self.saved_time_series_values_list = []
        self.output_path = output_path
        self.features = config['features']
        self.features_indexes = [features.index(feature) for feature in self.features]
        self.number = 0
        self.count = 0
        self.charts = []
        self.encoded_ids = {}

    def process_data(
            self,
            time_series_ids_list: List[int],
            time_series_times_list: List[np.ndarray],
            time_series_values_list: List[np.ndarray]
    ):
        left = self.total_number - len(self.saved_time_series_ids_list) - self.count
        if left > 0:
            self.saved_time_series_ids_list += time_series_ids_list[:left]
            self.saved_time_series_times_list += time_series_times_list[:left]
            self.saved_time_series_values_list += time_series_values_list[:left]

            if len(self.saved_time_series_ids_list) >= 10:  # for memory reasons
                self.visualize(private=True)
                self.saved_time_series_ids_list = []
                self.saved_time_series_times_list = []
                self.saved_time_series_values_list = []

    def visualize(self, private=False):
        for time_series_index, time_series_id in enumerate(self.saved_time_series_ids_list):
            self.encoded_ids[time_series_id] = self.count
            time_series_times = self.saved_time_series_times_list[time_series_index]
            time_series_values = self.saved_time_series_values_list[time_series_index]
            for feature_index, feature_label in enumerate(self.features):
                feature_index = self.features_indexes[feature_index]
                self.charts.append((
                    [(
                        time_series_values[feature_index, :],
                        time_series_times,
                        feature_label
                    )],
                    self.encoded_ids[time_series_id],
                    feature_label
                ))
            self.count += 1
            self.number += 1
            if self.number >= self.group_size:
                compare_multiple_lines_matrix(
                    False,
                    self.charts,
                    'Time series general look',
                    'values',
                    f'{self.output_path}/time_series_general_look_{self.count - self.number}-{self.count}',
                    ncols=len(self.features)
                )
                self.number = 0
                self.charts = []
        if self.number > 0 and not private:
            compare_multiple_lines_matrix(
                False,
                self.charts,
                'Time series general look',
                'values',
                f'{self.output_path}/time_series_general_look_{self.count - self.number}-{self.count}',
                ncols=len(self.features)
            )
        if not private:
            logger.info(f'Encoded chart ids: {self.encoded_ids}')


class ChartNullValues:
    def __init__(self, config: dict, features: List[str], output_path):
        self.features = features
        self.output_path = output_path
        self.null_values_dict = {}

        for feature in features:
            self.null_values_dict[feature] = {
                'total_values': 0,
                'total_null_values': 0,
                'time_series_with_null_values': set()
            }

    def process_data(
            self,
            time_series_ids_list: List[int],
            time_series_times_list: List[np.ndarray],
            time_series_values_list: List[np.ndarray]
    ):
        for time_series_index, time_series_id in enumerate(time_series_ids_list):
            time_series_values = time_series_values_list[time_series_index]
            for feature_index, feature_label in enumerate(self.features):
                feature_values = time_series_values[feature_index, :]
                null_values = np.count_nonzero(np.isnan(feature_values))
                self.null_values_dict[feature_label]['total_values'] += feature_values.shape[0]
                self.null_values_dict[feature_label]['total_null_values'] += null_values
                if null_values > 0:
                    self.null_values_dict[feature_label]['time_series_with_null_values'].add(time_series_id)

    def visualize(self):
        for feature in self.features:
            # make json serializable
            self.null_values_dict[feature]['time_series_with_null_values'] = len(list(self.null_values_dict[feature]['time_series_with_null_values']))

        print_pretty_json(self.null_values_dict)
        save_json(f'{self.output_path}/null_values', self.null_values_dict)

        plot_bar(
            False,
            f'{self.output_path}/null_values_absolute',
            f'Absolute null values',
            'features',
            'frequency',
            x=[feature for feature in self.features],
            height=[self.null_values_dict[feature]['total_null_values'] for feature in self.features]
        )

        plot_bar(
            False,
            f'{self.output_path}/null_values_percentage',
            f'Percentage null values',
            'features',
            'frequency',
            x=[feature for feature in self.features],
            height=[self.null_values_dict[feature]['total_null_values'] / self.null_values_dict[feature]['total_values'] for feature in self.features]
        )


class ChartMaxMinFeatures:
    def __init__(self, config: dict, features: List[str], output_path):
        self.features = features
        self.features_max_values = [None] * len(self.features)
        self.features_min_values = [None] * len(self.features)

    def process_data(
            self,
            time_series_ids_list: List[int],
            time_series_times_list: List[np.ndarray],
            time_series_values_list: List[np.ndarray]
    ):
        for time_series_index, time_series_id in enumerate(time_series_ids_list):
            time_series_values = time_series_values_list[time_series_index]
            for feature_index, feature_label in enumerate(self.features):
                feature_values = time_series_values[feature_index, :]
                max_value = np.max(feature_values)
                min_value = np.min(feature_values)

                if self.features_max_values[feature_index] is None:
                    self.features_max_values[feature_index] = max_value
                    self.features_min_values[feature_index] = min_value
                else:
                    self.features_max_values[feature_index] = max(max_value, self.features_max_values[feature_index])
                    self.features_min_values[feature_index] = min(min_value, self.features_min_values[feature_index])

    def visualize(self):
        max_values = {feature_label: self.features_max_values[feature_index]
                      for feature_index, feature_label in enumerate(self.features)}
        min_values = {feature_label: self.features_min_values[feature_index]
                      for feature_index, feature_label in enumerate(self.features)}
        logger.info(f'Max values: {max_values}')
        logger.info(f'Min values: {min_values}')


class ChartLengths:
    def __init__(self, config: dict, features: List[str], output_path):
        self.output_path = output_path
        self.lengths = []
        self.chart_params = config['chart_params']

    def process_data(
            self,
            time_series_ids_list: List[int],
            time_series_times_list: List[np.ndarray],
            time_series_values_list: List[np.ndarray]
    ):
        for time_series_index, time_series_id in enumerate(time_series_ids_list):
            time_series_times = time_series_times_list[time_series_index]
            self.lengths.append(time_series_times[-1] - time_series_times[0])

    def visualize(self):
        plot_hist(
            False,
            f'{self.output_path}/lengths',
            'Time series lengths',
            'lengths',
            'frequency',
            **{'x': self.lengths, **self.chart_params}
        )


class ChartDistributionValues:
    def __init__(self, config: dict, features: List[str], output_path):
        self.output_path = output_path
        self.total_number = config['number']
        self.chart_params = config['chart_params'] if 'chart_params' in config else None
        self.count = 0
        self.features = features + ['time']
        self.features_values = {}

    def process_data(
            self,
            time_series_ids_list: List[int],
            time_series_times_list: List[np.ndarray],
            time_series_values_list: List[np.ndarray]
    ):
        left = self.total_number - self.count
        if left > 0:
            self.count += len(time_series_values_list)
            for time_series_index, time_series_values in enumerate(time_series_values_list):
                for feature_index, feature_label in enumerate(self.features[:-1]):
                    feature_values = time_series_values[feature_index, :]
                    if feature_label in self.features_values:
                        self.features_values[feature_label] = np.concatenate((self.features_values[feature_label], feature_values))
                    else:
                        self.features_values[feature_label] = feature_values
                if 'time' in self.features_values:
                    self.features_values['time'] = np.concatenate((self.features_values['time'], time_series_times_list[time_series_index]))
                else:
                    self.features_values['time'] = time_series_times_list[time_series_index]

    def visualize(self):
        logger.info(f'Used number of values per feature: {[(feature, self.features_values[feature].shape[0]) for feature in self.features]}')

        for feature in self.features:
            if not np.isnan(self.features_values[feature]).all():
                plot_hist(
                    False,
                    f'{self.output_path}/distribution_values_{feature}',
                    '',
                    'values',
                    'frequency',
                    x=self.features_values[feature],
                    bins=30
                )


class ChartMeanStd:
    def __init__(self, config: dict, features: List[str], output_path):
        self.output_path = output_path
        self.computation = None
        self.features = features

    def process_data(
            self,
            time_series_ids_list: List[int],
            time_series_times_list: List[np.ndarray],
            time_series_values_list: List[np.ndarray]
    ):
        for time_series_values in time_series_values_list:
            if self.computation is None:
                self.computation = MeanStdIterativeComputation(time_series_values)
            else:
                self.computation.update(time_series_values)

    def visualize(self):
        mean, std = self.computation.finalize()
        logger.info(f'Means: {[(feature, mean[feature_index]) for feature_index, feature in enumerate(self.features)]}')
        logger.info(f'Stds: {[(feature, std[feature_index]) for feature_index, feature in enumerate(self.features)]}')


class ChartCorrelation:
    def __init__(self, config: dict, features: List[str], output_path):
        self.output_path = output_path
        self.features = features
        self.count = 0
        self.filtered_features = config['features'] if 'features' in config else None
        if self.filtered_features is not None:
            self.filtered_features_indexes = [features.index(feature) for feature in self.filtered_features]
            self.correlation_matrix = np.zeros((len(self.filtered_features), len(self.filtered_features)))
        else:
            self.filtered_features_indexes = None
            self.correlation_matrix = np.zeros((len(self.features), len(self.features)))

    def process_data(
            self,
            time_series_ids_list: List[int],
            time_series_times_list: List[np.ndarray],
            time_series_values_list: List[np.ndarray]
    ):
        self.count += len(time_series_values_list)
        for time_series_index, time_series_values in enumerate(time_series_values_list):
            if self.filtered_features_indexes is not None:
                time_series_values = np.take(time_series_values, self.filtered_features_indexes, axis=0)
            aux_matrix, _ = spearmanr(np.transpose(time_series_values))
            aux_matrix = np.nan_to_num(aux_matrix)  # Nan outputs when all values of a feature are constant
            self.correlation_matrix = np.add(self.correlation_matrix, aux_matrix)

    def visualize(self):
        self.correlation_matrix = np.divide(self.correlation_matrix, self.count, where=(self.correlation_matrix != 0))
        plot_correlation_matrix(
            False,
            self.correlation_matrix,
            self.filtered_features,
            self.output_path + '/compact_correlation'
        )


class ChartTimeDistanceBetweenSamples:
    def __init__(self, config: dict, features: List[str], output_path):
        self.output_path = output_path
        self.features = features
        self.time_distances_intervals = config['intervals']
        self.time_distances_counts = np.zeros(len(self.time_distances_intervals))

    def process_data(
            self,
            time_series_ids_list: List[int],
            time_series_times_list: List[np.ndarray],
            time_series_values_list: List[np.ndarray]
    ):
        for time_series_times in time_series_times_list:
            for time_index, time_value in enumerate(time_series_times[:-1]):
                time_distance = time_series_times[time_index + 1] - time_value
                if time_distance == 0:
                    position = 0
                else:
                    position = self.__binary_search(self.time_distances_intervals, 0, len(self.time_distances_intervals) - 1, time_distance)
                self.time_distances_counts[position] += 1

    def visualize(self):
        print(self.time_distances_counts)
        plot_bar(
            False,
            f'{self.output_path}/time_distances',
            'Time series time distances between samples',
            'time distances',
            'frequency',
            x=self.time_distances_intervals,
            height=self.time_distances_counts
        )

    def __binary_search(self, arr, low, high, x):
        # Check base case
        if high >= low:
            mid = (high + low) // 2
            # If element is present at the middle itself
            if arr[mid] == x:
                return mid
            # If element is smaller than mid, then it can only
            # be present in left subarray
            elif arr[mid] > x:
                return self.__binary_search(arr, low, mid - 1, x)
            # Else the element can only be present in right subarray
            else:
                return self.__binary_search(arr, mid + 1, high, x)
        else:
            # Element is not present in the array
            return high


def chart_autocorrelation(self, visualize: bool, config: dict, time_series_list: List[np.ndarray],
                          time_series_labels_list: List[str], output_path: str):
    number = config['number']
    if number == -1:
        chosen_time_series_list = time_series_list.copy()
    else:
        chosen_indexes = [random.randint(0, len(time_series_list) - 1) for _ in range(number)]
        chosen_time_series_list = []
        for chosen_index in chosen_indexes:
            chosen_time_series_list.append(time_series_list[chosen_index].copy())

    # normalize time series
    chosen_time_series_list, chosen_time_series_labels_list = self.preprocess([{'name': 'min_max_normalization'}],
                                                                              chosen_time_series_list)

    _type = config['type']

    if _type == 'compact':
        lines = []
        lags = config['lags']
        x = [str(index) for index in range(1, lags)]
        for time_series_index, time_series in enumerate(chosen_time_series_list):
            y = []
            for lag_index, lag in enumerate(range(1, lags)):
                aux1 = time_series[lag:, 0]
                aux2 = time_series[:(len(time_series) - lag), 0]
                if aux1.shape[0] > 0 < aux2.shape[0]:
                    autocorrelation, _ = spearmanr(aux1, aux2)
                    y.append(autocorrelation)
                else:
                    y.append(None)
            lines.append((y, x, 'time series ' + str(time_series_index)))

        compare_multiple_lines(visualize,
                               lines,
                               'autocorrelation',
                               'lags',
                               'Compact lag autocorrelation of target feature per time series',
                               output_path + '/lag_autocorrelation_compact_target'
                               )

        lags = config['lags']
        x = [str(index) for index in range(1, lags)]
        global_y = np.zeros((len(self.features), lags - 1))
        for time_series_index, time_series in enumerate(chosen_time_series_list):
            for lag_index, lag in enumerate(range(1, lags)):
                for feature_index in range(len(self.features)):
                    autocorrelation, _ = spearmanr(time_series[lag:, 0],
                                                   time_series[:(len(time_series) - lag), feature_index])
                    global_y[feature_index, lag_index] += autocorrelation
        global_y = np.divide(global_y, len(chosen_time_series_list))

        lines = [(global_y[feature_index], x, feature) for feature_index, feature in enumerate(self.features)]

        compare_multiple_lines(visualize,
                               lines,
                               'autocorrelation',
                               'lags',
                               'Compact lag autocorrelation with all features with the target one',
                               output_path + '/lag_autocorrelation_compact_features'
                               )

    elif _type == 'split':
        lags = config['lags']
        x = [str(index) for index in range(1, lags)]
        for time_series_index, time_series in enumerate(chosen_time_series_list):
            y = np.zeros((len(self.features), lags - 1))
            for lag_index, lag in enumerate(range(1, lags)):
                for feature_index in range(len(self.features)):
                    autocorrelation, _ = spearmanr(time_series[lag:, 0],
                                                   time_series[:(len(time_series) - lag), feature_index])
                    y[feature_index, lag_index] = autocorrelation
            lines = [(y[feature_index], x, feature) for feature_index, feature in enumerate(self.features)]

            compare_multiple_lines(visualize,
                                   lines,
                                   'autocorrelation',
                                   'lags',
                                   'Split lag autocorrelation with all features with the target one',
                                   output_path + '/lag_autocorrelation_split_features_time_series_' + str(
                                       time_series_index)
                                   )
    else:
        raise Exception('The type ' + _type + ' for chart autocorrelation does not exist')


def chart_preprocess(self, visualize: bool, config: dict, time_series_list: List[np.ndarray],
                     time_series_labels_list: List[str], output_path: str):
    time_series_list, time_series_labels_list = self.random_select_time_series(config['number'], time_series_list,
                                                                               time_series_labels_list)

    time_series_preprocessing_list = [time_series_list.copy()]
    preprocessing_labels = ['original']
    for preprocess_step_config in config['preprocessing_steps']:
        if preprocess_step_config['name'] not in ['filter_by_length', 'select']:
            preprocessing_labels.append(preprocess_step_config['name'])
            time_series_list, time_series_labels_list = self.preprocess([preprocess_step_config], time_series_list,
                                                                        time_series_labels_list)
            time_series_preprocessing_list.append(time_series_list.copy())

    _type = config['type']
    charts = []
    for time_series_index in range(len(time_series_preprocessing_list[0])):  # TODO refactor, error if empty
        for step_index, step_label in enumerate(preprocessing_labels):
            time_series = time_series_preprocessing_list[step_index][time_series_index]
            lines = []
            x = [index for index in range(time_series.shape[0])]
            if _type == 'all':
                for feature_index, feature_label in enumerate(self.features):
                    lines.append((time_series[:, feature_index], x, feature_label))
            elif _type == 'target':
                lines.append((time_series[:, 0], x, self.features[0]))
            else:
                raise Exception('Type ' + _type + ' not recognized')
            charts.append((lines, 'ts idx ' + str(time_series_labels_list[time_series_index]), step_label))

    # plot in groups
    groups_size = config['groups_size']
    for index in range(0, len(charts), groups_size * len(preprocessing_labels)):
        compare_multiple_lines_matrix(visualize,
                                      charts[index:(index + groups_size * len(preprocessing_labels))],
                                      'Examples of the preprocessing steps',
                                      'preprocessing step',
                                      output_path + '/preprocessing_examples_' + str(index) + '_' + _type,
                                      ncols=len(preprocessing_labels))
