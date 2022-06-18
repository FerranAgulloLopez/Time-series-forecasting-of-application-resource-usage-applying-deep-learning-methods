import logging
from typing import List
from bisect import bisect
from scipy.signal import argrelextrema

import numpy as np

logger = logging.getLogger(__name__)


def perform_preprocessing_steps(
        config: dict,
        features: List[str],
        time_series_ids_list: List[int],
        time_series_times_list: List[np.ndarray],
        time_series_values_list: List[np.ndarray]
) -> (List[str], List[int], List[np.ndarray], List[np.ndarray]):
    for step_config in config:
        name = step_config['name']
        features, time_series_ids_list, time_series_times_list, time_series_values_list = eval(name)(
            step_config,
            features,
            time_series_ids_list,
            time_series_times_list,
            time_series_values_list
        )

    return features, time_series_ids_list, time_series_times_list, time_series_values_list


def filter_by_length(config: dict, time_series_list: List[np.ndarray], time_series_labels_list: List[str]):
    lengths = [len(time_series) for time_series in time_series_list]

    max_length = config['max']
    min_length = config['min']
    if max_length == -1: max_length = max(lengths)
    if min_length == -1: min_length = min(lengths)

    final_time_series_list = []
    final_time_series_labels_list = []
    for index, time_series in enumerate(time_series_list):
        time_series_list_length = time_series.shape[0]
        if max_length >= time_series_list_length >= min_length:
            final_time_series_list.append(time_series_list[index])
            final_time_series_labels_list.append(time_series_labels_list[index])
        else:
            print('Time series filtered by length:', time_series_labels_list[index])
    return final_time_series_list, final_time_series_labels_list  # TODO check if necessary


def delete_time_series(
        config: dict,
        features: List[str],
        time_series_ids_list: List[int],
        time_series_times_list: List[np.ndarray],
        time_series_values_list: List[np.ndarray]
) -> (List[int], List[np.ndarray], List[np.ndarray]):
    ids_to_delete = set(config['ids'])
    ids_to_delete_indexes = []
    for time_series_index, time_series_id in enumerate(time_series_ids_list):
        if time_series_id in ids_to_delete:
            ids_to_delete_indexes.append(time_series_index)
    for index in sorted(ids_to_delete_indexes, reverse=True):
        del time_series_ids_list[index]
        del time_series_times_list[index]
        del time_series_values_list[index]
    return features, time_series_ids_list, time_series_times_list, time_series_values_list  # TODO check if necessary


def delete_time_series_per_length(
        config: dict,
        features: List[str],
        time_series_ids_list: List[int],
        time_series_times_list: List[np.ndarray],
        time_series_values_list: List[np.ndarray]
) -> (List[int], List[np.ndarray], List[np.ndarray]):
    max = config['max'] if 'max' in config else None
    min = config['min'] if 'min' in config else None
    ids_to_delete_indexes = []
    for time_series_index, time_series_times in enumerate(time_series_times_list):
        length = time_series_times[-1] - time_series_times[0]
        if max and length > max:
            ids_to_delete_indexes.append(time_series_index)
        elif min and length < min:
            ids_to_delete_indexes.append(time_series_index)
    for index in sorted(ids_to_delete_indexes, reverse=True):
        del time_series_ids_list[index]
        del time_series_times_list[index]
        del time_series_values_list[index]
    return features, time_series_ids_list, time_series_times_list, time_series_values_list  # TODO check if necessary


def delete_time_series_per_number_of_samples(
        config: dict,
        features: List[str],
        time_series_ids_list: List[int],
        time_series_times_list: List[np.ndarray],
        time_series_values_list: List[np.ndarray]
) -> (List[int], List[np.ndarray], List[np.ndarray]):
    max = config['max'] if 'max' in config else None
    min = config['min'] if 'min' in config else None
    ids_to_delete_indexes = []
    for time_series_index, time_series_times in enumerate(time_series_times_list):
        length = time_series_times.shape[0]
        if max and length > max:
            ids_to_delete_indexes.append(time_series_index)
        elif min and length < min:
            ids_to_delete_indexes.append(time_series_index)
    for index in sorted(ids_to_delete_indexes, reverse=True):
        del time_series_ids_list[index]
        del time_series_times_list[index]
        del time_series_values_list[index]
    return features, time_series_ids_list, time_series_times_list, time_series_values_list  # TODO check if necessary


def delete_time_series_with_many_nan_values(
        config: dict,
        features: List[str],
        time_series_ids_list: List[int],
        time_series_times_list: List[np.ndarray],
        time_series_values_list: List[np.ndarray]
) -> (List[int], List[np.ndarray], List[np.ndarray]):
    features_to_check = config['features']
    features_to_check_indexes = [features.index(feature['name']) for feature in features_to_check]
    features_to_check_percentages = [feature['max_nans_percentage'] for feature in features_to_check]
    ids_to_delete_indexes = []
    for time_series_index, time_series_values in enumerate(time_series_values_list):
        for index, feature in enumerate(features_to_check):
            feature_index = features_to_check_indexes[index]
            max_percentage = features_to_check_percentages[index]
            nan_values_percentage = np.count_nonzero(np.isnan(time_series_values[feature_index, :])) / time_series_values.shape[1]
            if nan_values_percentage > max_percentage:
                ids_to_delete_indexes.append(time_series_index)
                break
    for index in sorted(ids_to_delete_indexes, reverse=True):
        del time_series_ids_list[index]
        del time_series_times_list[index]
        del time_series_values_list[index]
    return features, time_series_ids_list, time_series_times_list, time_series_values_list  # TODO check if necessary


def delete_samples_with_specific_values(
        config: dict,
        features: List[str],
        time_series_ids_list: List[int],
        time_series_times_list: List[np.ndarray],
        time_series_values_list: List[np.ndarray]
) -> (List[int], List[np.ndarray], List[np.ndarray]):
    features_to_check = config['features']
    features_to_check_indexes = [features.index(feature['name']) for feature in features_to_check]
    for time_series_index, time_series_id in enumerate(time_series_ids_list):
        time_series_values = time_series_values_list[time_series_index]
        for index, feature in enumerate(features_to_check):
            value_to_remove = feature['value_to_remove']
            feature_index = features_to_check_indexes[index]
            time_series_values = time_series_values[:, time_series_values[feature_index, :] != value_to_remove]

    return features, time_series_ids_list, time_series_times_list, time_series_values_list  # TODO check if necessary


def replace_nan_with_previous(
        config: dict,
        features: List[str],
        time_series_ids_list: List[int],
        time_series_times_list: List[np.ndarray],
        time_series_values_list: List[np.ndarray]
) -> (List[int], List[np.ndarray], List[np.ndarray]):
    features_to_check = config['features']
    features_to_check_indexes = [features.index(feature['name']) for feature in features_to_check]
    for time_series_index, time_series_id in enumerate(time_series_ids_list):
        time_series_values = time_series_values_list[time_series_index]
        for index, feature in enumerate(features_to_check):
            feature_index = features_to_check_indexes[index]
            for value_index in range(time_series_values.shape[1]):
                if np.isnan(time_series_values[feature_index, value_index]):
                    if value_index == 0:
                        time_series_values[feature_index, value_index] = 0
                    else:
                        time_series_values[feature_index, value_index] = time_series_values[feature_index, value_index - 1]

    return features, time_series_ids_list, time_series_times_list, time_series_values_list  # TODO check if necessary


def replace_outliers_with_previous(
        config: dict,
        features: List[str],
        time_series_ids_list: List[int],
        time_series_times_list: List[np.ndarray],
        time_series_values_list: List[np.ndarray]
) -> (List[int], List[np.ndarray], List[np.ndarray]):
    features_to_check = config['features']
    features_to_check_indexes = [features.index(feature['name']) for feature in features_to_check]
    for time_series_index, time_series_id in enumerate(time_series_ids_list):
        time_series_values = time_series_values_list[time_series_index]
        for index, feature in enumerate(features_to_check):
            feature_index = features_to_check_indexes[index]
            if 'max' in feature and time_series_values[feature_index, 0] > feature['max']:
                time_series_values[feature_index, 0] = 0
            if 'min' in feature and time_series_values[feature_index, 0] < feature['min']:
                time_series_values[feature_index, 0] = 0
            for value_index in range(1, time_series_values.shape[1]):
                if 'max' in feature and time_series_values[feature_index, value_index] > feature['max']:
                    time_series_values[feature_index, value_index] = time_series_values[feature_index, value_index - 1]
                if 'min' in feature and time_series_values[feature_index, value_index] < feature['min']:
                    time_series_values[feature_index, value_index] = time_series_values[feature_index, value_index - 1]

    return features, time_series_ids_list, time_series_times_list, time_series_values_list  # TODO check if necessary


def delete_close_samples(
        config: dict,
        features: List[str],
        time_series_ids_list: List[int],
        time_series_times_list: List[np.ndarray],
        time_series_values_list: List[np.ndarray]
) -> (List[int], List[np.ndarray], List[np.ndarray]):
    _range = config['range']
    total_samples = 0
    removed_samples = 0
    for time_series_index, time_series_id in enumerate(time_series_ids_list):
        time_series_times = time_series_times_list[time_series_index]
        time_series_values = time_series_values_list[time_series_index]
        total_samples += time_series_times.shape[0]
        time_index = 1
        while time_index < time_series_times.shape[0]:
            if abs(time_series_times[time_index] - time_series_times[time_index - 1]) < _range:
                time_series_times = np.delete(time_series_times, time_index - 1, axis=0)
                time_series_values[:, time_index] = np.mean(time_series_values[:, (time_index - 1):(time_index + 1)], axis=1)
                time_series_values = np.delete(time_series_values, time_index - 1, axis=1)
                removed_samples += 1
            else:
                time_index += 1

    return features, time_series_ids_list, time_series_times_list, time_series_values_list  # TODO check if necessary


def log_transformation(
        config: dict,
        features: List[str],
        time_series_ids_list: List[int],
        time_series_times_list: List[np.ndarray],
        time_series_values_list: List[np.ndarray]
) -> (List[int], List[np.ndarray], List[np.ndarray]):
    _min = config['min'] if 'min' in config else None
    for time_series_index, time_series_id in enumerate(time_series_ids_list):
        for feature_index in range(time_series_values_list[time_series_index].shape[0]):
            feature_values = time_series_values_list[time_series_index][feature_index]
            transformed_feature_values = np.log(feature_values + 0.00000000001)
            if _min is not None:
                transformed_feature_values = np.where(transformed_feature_values > _min, transformed_feature_values, _min)
            time_series_values_list[time_series_index][feature_index] = transformed_feature_values
    return features, time_series_ids_list, time_series_times_list, time_series_values_list  # TODO check if necessary


def sum(
        config: dict,
        features: List[str],
        time_series_ids_list: List[int],
        time_series_times_list: List[np.ndarray],
        time_series_values_list: List[np.ndarray]
) -> (List[int], List[np.ndarray], List[np.ndarray]):
    values = config['values']
    for time_series_index, time_series_id in enumerate(time_series_ids_list):
        time_series_values = time_series_values_list[time_series_index]
        values_aux = np.ones((time_series_values.shape[0], time_series_values.shape[1])) * np.transpose(np.asarray([values]))
        time_series_values_list[time_series_index] = time_series_values + values_aux
    return features, time_series_ids_list, time_series_times_list, time_series_values_list  # TODO check if necessary


def select_features(config: dict, time_series_list: List[np.ndarray], time_series_labels_list: List[str]):
    features = config['features']
    final_time_series_list = []
    for index in range(len(time_series_list)):
        final_time_series_list.append(np.take(time_series_list[index], features, axis=1))
    return final_time_series_list, time_series_labels_list  # TODO check if necessary


def select_feature(
        config: dict,
        features: List[str],
        time_series_ids_list: List[int],
        time_series_times_list: List[np.ndarray],
        time_series_values_list: List[np.ndarray]
) -> (List[int], List[np.ndarray], List[np.ndarray]):
    feature_index = features.index(config['feature'])
    for time_series_index, time_series_id in enumerate(time_series_ids_list):
        time_series_values = time_series_values_list[time_series_index]
        time_series_values_list[time_series_index] = time_series_values[feature_index, :]
    return features, time_series_ids_list, time_series_times_list, time_series_values_list  # TODO check if necessary


def select_feature_2(
        config: dict,
        features: List[str],
        time_series_ids_list: List[int],
        time_series_times_list: List[np.ndarray],
        time_series_values_list: List[np.ndarray]
) -> (List[int], List[np.ndarray], List[np.ndarray]):
    feature_index = features.index(config['feature'])
    for time_series_index, time_series_id in enumerate(time_series_ids_list):
        time_series_values = time_series_values_list[time_series_index]
        time_series_values_list[time_series_index] = np.expand_dims(time_series_values[feature_index, :], axis=0)
    features = [features[feature_index]]
    return features, time_series_ids_list, time_series_times_list, time_series_values_list  # TODO check if necessary


def delete_features(
        config: dict,
        features: List[str],
        time_series_ids_list: List[int],
        time_series_times_list: List[np.ndarray],
        time_series_values_list: List[np.ndarray]
) -> (List[int], List[np.ndarray], List[np.ndarray]):
    features_labels = config['features']
    features_indexes = [features.index(feature) for feature in features_labels]
    for time_series_index, time_series_id in enumerate(time_series_ids_list):
        time_series_values = time_series_values_list[time_series_index]
        time_series_values_list[time_series_index] = np.delete(time_series_values, features_indexes, axis=0)
    features = [feature for feature in features if feature not in features_labels]
    return features, time_series_ids_list, time_series_times_list, time_series_values_list  # TODO check if necessary


def max_moving_average(config: dict, time_series_list: List[np.ndarray], time_series_labels_list: List[str]):
    window_size = config['window_size']
    for index_time_series, time_series in enumerate(time_series_list):
        final_time_series = np.zeros(time_series.shape)
        final_time_series[:window_size] = time_series[:window_size]
        for index_position in range(window_size, final_time_series.shape[0]):
            final_time_series[index_position] = np.max(time_series[(index_position - window_size):index_position],
                                                       axis=0)
        time_series_list[index_time_series] = final_time_series
    return time_series_list, time_series_labels_list  # TODO check if necessary


def max_moving_average_pre(
        config: dict,
        features: List[str],
        time_series_ids_list: List[int],
        time_series_times_list: List[np.ndarray],
        time_series_values_list: List[np.ndarray]
) -> (List[int], List[np.ndarray], List[np.ndarray]):
    window_size = config['window_size']
    for time_series_index, time_series_id in enumerate(time_series_ids_list):
        time_series_values = time_series_values_list[time_series_index]
        for time_position in range(time_series_values.shape[0] - window_size):  # TODO only works with one feature arrays
            time_series_values[time_position] = np.max(
                time_series_values[time_position:(time_position + window_size)]
            )
    return features, time_series_ids_list, time_series_times_list, time_series_values_list  # TODO check if necessary


def mean_max_moving_average_pre(
        config: dict,
        features: List[str],
        time_series_ids_list: List[int],
        time_series_times_list: List[np.ndarray],
        time_series_values_list: List[np.ndarray]
) -> (List[int], List[np.ndarray], List[np.ndarray]):
    window_size = config['window_size']
    for time_series_index, time_series_id in enumerate(time_series_ids_list):
        time_series_values = time_series_values_list[time_series_index]
        for time_position in range(time_series_values.shape[0] - window_size):  # TODO only works with one feature arrays
            max_value = np.max(
                time_series_values[time_position:(time_position + window_size)]
            )
            if time_series_values[time_position] != max_value:
                time_series_values[time_position] = np.mean(
                    time_series_values[time_position:(time_position + window_size)]
                )
    return features, time_series_ids_list, time_series_times_list, time_series_values_list  # TODO check if necessary


def max_moving_average_inverse(
        config: dict,
        features: List[str],
        time_series_ids_list: List[int],
        time_series_times_list: List[np.ndarray],
        time_series_values_list: List[np.ndarray]
) -> (List[int], List[np.ndarray], List[np.ndarray]):
    window_size = config['window_size']
    for time_series_index, time_series_id in enumerate(time_series_ids_list):
        time_series_values = time_series_values_list[time_series_index]
        if len(time_series_values.shape) > 1:
            for time_position in range(time_series_values.shape[1] - 1, window_size, -1):
                time_series_values[:, time_position] = np.max(
                    time_series_values[:, (time_position - window_size + 1):(time_position + 1)],
                    axis=1
                )
        else:
            for time_position in range(time_series_values.shape[0] - 1, window_size, -1):
                time_series_values[time_position] = np.max(
                    time_series_values[(time_position - window_size + 1):(time_position + 1)]
                )
    return features, time_series_ids_list, time_series_times_list, time_series_values_list  # TODO check if necessary


def mean_max_moving_average_inverse(
        config: dict,
        features: List[str],
        time_series_ids_list: List[int],
        time_series_times_list: List[np.ndarray],
        time_series_values_list: List[np.ndarray]
) -> (List[int], List[np.ndarray], List[np.ndarray]):
    window_size = config['window_size']
    for time_series_index, time_series_id in enumerate(time_series_ids_list):
        time_series_values = time_series_values_list[time_series_index]
        if len(time_series_values.shape) > 1:
            raise NotImplementedError()  # not trivial to do it, with which feature do you the conditional?
        else:
            for time_position in range(time_series_values.shape[0] - 1, window_size, -1):
                max_value = np.max(
                    time_series_values[(time_position - window_size + 1):(time_position + 1)]
                )
                if time_series_values[time_position] != max_value:
                    time_series_values[time_position] = np.mean(
                        time_series_values[(time_position - window_size + 1):(time_position + 1)]
                    )
    return features, time_series_ids_list, time_series_times_list, time_series_values_list  # TODO check if necessary


def mean_max_moving_average_inverse_strange(
        config: dict,
        features: List[str],
        time_series_ids_list: List[int],
        time_series_times_list: List[np.ndarray],
        time_series_values_list: List[np.ndarray]
) -> (List[int], List[np.ndarray], List[np.ndarray]):
    window_size = config['window_size']
    for time_series_index, time_series_id in enumerate(time_series_ids_list):
        time_series_values = time_series_values_list[time_series_index]
        if len(time_series_values.shape) > 1:
            raise NotImplementedError()  # not trivial to do it, with which feature do you the conditional?
        else:
            for time_position in range(time_series_values.shape[0] - 1, window_size, -1):
                window = time_series_values[(time_position - window_size + 1):(time_position + 1)]
                value = time_series_values[time_position]
                max_value_index = np.argmax(window)
                max_value = window[max_value_index]
                difference = max_value - value
                weight = max_value_index / window_size
                final_value = value + weight * difference
                if final_value > value:
                    time_series_values[time_position] = final_value
    return features, time_series_ids_list, time_series_times_list, time_series_values_list  # TODO check if necessary


def mean_max_moving_average_inverse_strange_2(
        config: dict,
        features: List[str],
        time_series_ids_list: List[int],
        time_series_times_list: List[np.ndarray],
        time_series_values_list: List[np.ndarray]
) -> (List[int], List[np.ndarray], List[np.ndarray]):
    window_size = config['window_size']
    for time_series_index, time_series_id in enumerate(time_series_ids_list):
        time_series_values = time_series_values_list[time_series_index]
        if len(time_series_values.shape) > 1:
            raise NotImplementedError()  # not trivial to do it, with which feature do you the conditional?
        else:
            for time_position in range(time_series_values.shape[0] - 1, window_size, -1):
                window = time_series_values[(time_position - window_size + 1):(time_position + 1)]
                close_local_maximum_index = window.shape[0] - 1
                while close_local_maximum_index > 0 and window[close_local_maximum_index] < window[close_local_maximum_index - 1]:
                    close_local_maximum_index -= 1
                time_series_values[time_position] = np.mean(window[close_local_maximum_index:])
    return features, time_series_ids_list, time_series_times_list, time_series_values_list  # TODO check if necessary


def mean_max_moving_average_inverse_strange_3(
        config: dict,
        features: List[str],
        time_series_ids_list: List[int],
        time_series_times_list: List[np.ndarray],
        time_series_values_list: List[np.ndarray]
) -> (List[int], List[np.ndarray], List[np.ndarray]):
    window_size = config['window_size']
    for time_series_index, time_series_id in enumerate(time_series_ids_list):
        time_series_values = time_series_values_list[time_series_index]
        if len(time_series_values.shape) > 1:
            for time_position in range(time_series_values.shape[1] - 1, window_size, -1):
                values = time_series_values[:, time_position]
                window = time_series_values[:, (time_position - window_size + 1):(time_position + 1)]
                max_value_indexes = np.argmax(window, axis=1)
                mean_values = [np.mean([values[index], window[index, max_value_indexes[index]]]) for index in range(window.shape[0])]
                final_values = [values[index] if values[index] > mean_values[index] else mean_values[index] for index in range(window.shape[0])]
                time_series_values[:, time_position] = final_values
        else:
            for time_position in range(time_series_values.shape[0] - 1, window_size, -1):
                value = time_series_values[time_position]
                window = time_series_values[(time_position - window_size + 1):(time_position + 1)]
                max_value_index = np.argmax(window)
                mean_value = np.mean(window[max_value_index:])
                # mean_value = np.mean([value, window[max_value_index]])
                if mean_value > value:
                    time_series_values[time_position] = mean_value
    return features, time_series_ids_list, time_series_times_list, time_series_values_list  # TODO check if necessary


def mean_max_moving_average_inverse_strange_5(
        config: dict,
        features: List[str],
        time_series_ids_list: List[int],
        time_series_times_list: List[np.ndarray],
        time_series_values_list: List[np.ndarray]
) -> (List[int], List[np.ndarray], List[np.ndarray]):
    window_size = config['window_size']
    for time_series_index, time_series_id in enumerate(time_series_ids_list):
        time_series_values = time_series_values_list[time_series_index]
        if len(time_series_values.shape) > 1:
            for time_position in range(time_series_values.shape[1] - 1, window_size, -1):
                values = time_series_values[:, time_position]
                window = time_series_values[:, (time_position - window_size + 1):(time_position + 1)]
                max_value_indexes = np.argmax(window, axis=1)
                difference_values = [window[index, max_value_indexes[index]] - values[index] for index in range(window.shape[0])]
                weight_values = [max_value_indexes[index] / window_size for index in range(window.shape[0])]
                final_values = [max(values[index], values[index] + difference_values[index] * weight_values[index]) for index in range(window.shape[0])]
                time_series_values[:, time_position] = final_values
        else:
            for time_position in range(time_series_values.shape[0] - 1, window_size, -1):
                value = time_series_values[time_position]
                window = time_series_values[(time_position - window_size + 1):(time_position + 1)]
                max_value_index = np.argmax(window)
                max_value = window[max_value_index]
                difference = max_value - value
                weight = max_value_index / window_size
                final_value = value + weight * difference
                if final_value > value:
                    time_series_values[time_position] = final_value
    return features, time_series_ids_list, time_series_times_list, time_series_values_list  # TODO check if necessary


def mean_max_moving_average_pre_strange_3(
        config: dict,
        features: List[str],
        time_series_ids_list: List[int],
        time_series_times_list: List[np.ndarray],
        time_series_values_list: List[np.ndarray]
) -> (List[int], List[np.ndarray], List[np.ndarray]):
    window_size = config['window_size']
    for time_series_index, time_series_id in enumerate(time_series_ids_list):
        time_series_values = time_series_values_list[time_series_index]
        if len(time_series_values.shape) > 1:
            for time_position in range(window_size, time_series_values.shape[1]):
                values = time_series_values[:, time_position]
                window = time_series_values[:, (time_position - window_size + 1):(time_position + 1)]
                max_value_indexes = np.argmax(window, axis=1)
                mean_values = [np.mean(window[index, max_value_indexes[index]:]) for index in range(window.shape[0])]
                final_values = [values[index] if values[index] > mean_values[index] else mean_values[index] for index in range(window.shape[0])]
                time_series_values[:, time_position] = final_values
        else:
            for time_position in range(window_size, time_series_values.shape[0]):
                value = time_series_values[time_position]
                window = time_series_values[(time_position - window_size + 1):(time_position + 1)]
                max_value_index = np.argmax(window)
                mean_value = np.mean(window[max_value_index:])
                if mean_value > value:
                    time_series_values[time_position] = mean_value
    return features, time_series_ids_list, time_series_times_list, time_series_values_list  # TODO check if necessary


def mean_max_moving_average_inverse_strange_4(
        config: dict,
        features: List[str],
        time_series_ids_list: List[int],
        time_series_times_list: List[np.ndarray],
        time_series_values_list: List[np.ndarray]
) -> (List[int], List[np.ndarray], List[np.ndarray]):
    window_size = config['window_size']
    for time_series_index, time_series_id in enumerate(time_series_ids_list):
        time_series_values = time_series_values_list[time_series_index]
        if len(time_series_values.shape) > 1:
            raise NotImplementedError()  # not trivial to do it, with which feature do you the conditional?
        else:
            for time_position in range(time_series_values.shape[0] - 1, window_size, -1):
                value = time_series_values[time_position]
                window = time_series_values[(time_position - window_size + 1):(time_position + 1)]
                max_value_index = np.argmax(window)
                mean_value = np.mean(window[max_value_index:])
                # mean_value = np.mean([value, window[max_value_index]])
                if mean_value > value:
                    time_series_values[time_position] = mean_value
    return features, time_series_ids_list, time_series_times_list, time_series_values_list  # TODO check if necessary


def mean_moving_average_inverse(
        config: dict,
        features: List[str],
        time_series_ids_list: List[int],
        time_series_times_list: List[np.ndarray],
        time_series_values_list: List[np.ndarray]
) -> (List[int], List[np.ndarray], List[np.ndarray]):
    window_size = config['window_size']
    for time_series_index, time_series_id in enumerate(time_series_ids_list):
        time_series_values = time_series_values_list[time_series_index]
        if len(time_series_values.shape) > 1:
            for time_position in range(time_series_values.shape[1] - 1, window_size, -1):
                time_series_values[:, time_position] = np.mean(
                    time_series_values[:, (time_position - window_size + 1):(time_position + 1)],
                    axis=1
                )
        else:
            for time_position in range(time_series_values.shape[0] - 1, window_size, -1):
                time_series_values[time_position] = np.mean(
                    time_series_values[(time_position - window_size + 1):(time_position + 1)]
                )
    return features, time_series_ids_list, time_series_times_list, time_series_values_list  # TODO check if necessary


def transpose(
        config: dict,
        features: List[str],
        time_series_ids_list: List[int],
        time_series_times_list: List[np.ndarray],
        time_series_values_list: List[np.ndarray]
) -> (List[int], List[np.ndarray], List[np.ndarray]):
    window_size = config['window_size']
    for time_series_index, time_series_id in enumerate(time_series_ids_list):
        time_series_values = time_series_values_list[time_series_index]
        if len(time_series_values.shape) > 1:
            for time_position in range(time_series_values.shape[1] - window_size):
                time_series_values[:, time_position] = time_series_values[:, (time_position + window_size)]
            time_series_values[:, (time_series_values.shape[1] - window_size):time_series_values.shape[1]] = np.ones((time_series_values.shape[0], window_size)) * np.transpose([time_series_values[:, (time_series_values.shape[1] - window_size)]])
        else:
            for time_position in range(time_series_values.shape[0] - window_size):
                time_series_values[time_position] = time_series_values[time_position + window_size]
            time_series_values[(time_series_values.shape[0] - window_size):time_series_values.shape[0]] = np.transpose([time_series_values[time_series_values.shape[0] - window_size]])
    return features, time_series_ids_list, time_series_times_list, time_series_values_list  # TODO check if necessary


def mean_moving_average_pre(
        config: dict,
        features: List[str],
        time_series_ids_list: List[int],
        time_series_times_list: List[np.ndarray],
        time_series_values_list: List[np.ndarray]
) -> (List[int], List[np.ndarray], List[np.ndarray]):
    window_size = config['window_size']
    for time_series_index, time_series_id in enumerate(time_series_ids_list):
        time_series_values = time_series_values_list[time_series_index]
        if len(time_series_values.shape) > 1:
            for time_position in range(time_series_values.shape[1] - window_size):  # TODO only works with one feature arrays
                time_series_values[:, time_position] = np.mean(
                    time_series_values[:, time_position:(time_position + window_size)],
                    axis=1
                )
        else:
            for time_position in range(time_series_values.shape[0] - window_size):
                time_series_values[time_position] = np.mean(
                    time_series_values[time_position:(time_position + window_size)]
                )
    return features, time_series_ids_list, time_series_times_list, time_series_values_list  # TODO check if necessary


def elevate(
        config: dict,
        features: List[str],
        time_series_ids_list: List[int],
        time_series_times_list: List[np.ndarray],
        time_series_values_list: List[np.ndarray]
) -> (List[int], List[np.ndarray], List[np.ndarray]):
    rate = config['rate']
    for time_series_index, time_series_id in enumerate(time_series_ids_list):
        time_series_values = time_series_values_list[time_series_index]
        if len(time_series_values.shape) > 1:
            raise NotImplementedError()
        else:
            for time_position in range(time_series_values.shape[0]):
                time_series_values[time_position] += time_series_values[time_position] * rate * np.sign(time_series_values[time_position])
    return features, time_series_ids_list, time_series_times_list, time_series_values_list  # TODO check if necessary


def intervals(
        config: dict,
        features: List[str],
        time_series_ids_list: List[int],
        time_series_times_list: List[np.ndarray],
        time_series_values_list: List[np.ndarray]
) -> (List[int], List[np.ndarray], List[np.ndarray]):
    intervals = config['intervals']
    for time_series_index, time_series_id in enumerate(time_series_ids_list):
        time_series_values = time_series_values_list[time_series_index]
        for time_position in range(time_series_values.shape[0]):
            value = time_series_values[time_position]
            interval_index = bisect(intervals, value)
            time_series_values[time_position] = intervals[interval_index]
    return features, time_series_ids_list, time_series_times_list, time_series_values_list  # TODO check if necessary


# TODO test mean exponential
def mean_weighted_moving_average(config: dict, time_series_list: List[np.ndarray], time_series_labels_list: List[str]):
    window_size = config['window_size']
    weights = np.arange(1, window_size + 1)
    for index_time_series, time_series in enumerate(time_series_list):
        final_time_series = np.zeros(time_series.shape)
        final_time_series[:window_size] = time_series[:window_size]
        for index_position in range(window_size, final_time_series.shape[0]):
            final_time_series[index_position] = np.dot(
                time_series[(index_position - window_size + 1):(index_position + 1)].T, weights) / weights.sum()
        time_series_list[index_time_series] = final_time_series
    return time_series_list, time_series_labels_list  # TODO check if necessary


def differencing(config: dict, time_series_list: List[np.ndarray], time_series_labels_list: List[str]):
    for index_time_series, time_series in enumerate(time_series_list):
        final_time_series = np.zeros((time_series.shape[0] - 1, time_series.shape[1]))
        for index_position in range(final_time_series.shape[0]):
            final_time_series[index_position] = time_series[index_position + 1] - time_series[index_position]
        time_series_list[index_time_series] = final_time_series
    return time_series_list, time_series_labels_list  # TODO check if necessary
