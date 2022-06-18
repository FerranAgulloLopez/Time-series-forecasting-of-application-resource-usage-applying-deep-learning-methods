import math
import random
from typing import List

import numpy as np


def shuffle_in_unison(*lists: List[np.ndarray]):
    rng_state = np.random.get_state()
    for _list in lists:
        np.random.shuffle(_list)
        np.random.set_state(rng_state)


def randomly_split_list_values_in_three(_list: List, size_split_1, size_split_2):
    if 0 >= (size_split_1 + size_split_2) >= 1 or size_split_1 <= 0 or size_split_2 <= 0:
        raise Exception('Split sizes not correct, they must be over 0 and sum less than 1')
    number_samples_split_1 = math.floor(len(_list) * size_split_1)
    number_samples_split_2 = math.floor(len(_list) * size_split_2)
    number_samples_split_3 = len(_list) - number_samples_split_1 - number_samples_split_2

    random.shuffle(_list)  # TODO check real array is not shuffled
    samples_split_1 = _list[:number_samples_split_1]
    samples_split_2 = _list[number_samples_split_1:(number_samples_split_1 + number_samples_split_2)]
    samples_split_3 = _list[-number_samples_split_3:]

    return samples_split_1, samples_split_2, samples_split_3


def assign_split_values_by_label(
        split_labels,
        labels_time_series_dict: dict,  # dict[str, int]
        values_time_series_list: List[np.ndarray],
        target_time_series_list: List[np.ndarray]
        ) -> (List[str], List[np.ndarray], List[np.ndarray]):

    split_labels_time_series_list = []
    split_values_time_series_list = []
    split_target_time_series_list = []

    for label in split_labels:
        if label not in labels_time_series_dict:
            raise Exception(
                'Error when splitting, the time series ' + str(label) + ' does not exist in the original dataset')
        index = labels_time_series_dict[label]
        split_labels_time_series_list.append(label)
        split_values_time_series_list.append(values_time_series_list[index])
        split_target_time_series_list.append(target_time_series_list[index])

    return split_labels_time_series_list, split_values_time_series_list, split_target_time_series_list


# from https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance
# modified to allow multiple values and dimensions
class MeanStdIterativeComputation:
    def __init__(self, values):
        self.existingAggregate = [values.shape[1]] * values.shape[0], np.mean(values, axis=1), (np.var(values, axis=1) * [values.shape[1]] * values.shape[0]) / 2

    def update(self, newValues):
        (count, mean, M2) = self.existingAggregate
        count += np.asarray([newValues.shape[1]] * newValues.shape[0])
        delta = np.subtract(newValues, np.ones((newValues.shape[0], newValues.shape[1])) * np.transpose(np.asarray([mean])))
        mean += np.sum(np.divide(delta, np.transpose([count])), axis=1)
        delta2 = np.subtract(newValues, np.ones((newValues.shape[0], newValues.shape[1])) * np.transpose(np.asarray([mean])))
        M2 += np.sum(delta * delta2, axis=1)
        self.existingAggregate = (count, mean, M2)

    def finalize(self):
        (count, mean, M2) = self.existingAggregate
        (mean, variance, sampleVariance) = (mean, M2 / count, M2 / (count - 1))
        return mean, np.sqrt(variance)
