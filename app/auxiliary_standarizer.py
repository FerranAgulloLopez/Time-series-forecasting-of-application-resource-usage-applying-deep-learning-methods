from os import listdir
from os.path import isfile, exists, join

import numpy as np

from app.auxiliary_files.other_methods.util_functions import load_csv

INPUT_PATH = '../data/alibaba_2018/data/third_preprocess_compressed'
OUTPUT_PATH = '../data/alibaba_2018/data/third_preprocess_compressed_standarized'
TIME_FEATURE_INDEX = 0  # wont be standarized


# from https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance
# modified to allow multiple values and dimensions
def initialize(values):
    return [values.shape[1]] * values.shape[0], np.mean(values, axis=1), (
                np.var(values, axis=1) * [values.shape[1]] * values.shape[0]) / 2


def update_array_dims(existingAggregate, newValues):
    (count, mean, M2) = existingAggregate
    count += np.asarray([newValues.shape[1]] * newValues.shape[0])
    delta = np.subtract(newValues, np.ones((newValues.shape[0], newValues.shape[1])) * np.transpose(np.asarray([mean])))
    mean += np.sum(np.divide(delta, np.transpose([count])), axis=1)
    delta2 = np.subtract(newValues,
                         np.ones((newValues.shape[0], newValues.shape[1])) * np.transpose(np.asarray([mean])))
    M2 += np.sum(delta * delta2, axis=1)
    return count, mean, M2


def finalize_array_dims(existingAggregate):
    (count, mean, M2) = existingAggregate
    (mean, variance, sampleVariance) = (mean, M2 / count, M2 / (count - 1))
    return mean, variance, sampleVariance


input_files = [
    filename.replace('.npy', '') for filename in listdir(INPUT_PATH)
    if isfile(join(INPUT_PATH, filename))
    and filename.endswith('.npy')
]

print('Check mean and std for each feature')
header = load_csv(f'{INPUT_PATH}/header.csv')[0]
iterative_values = None
count = 0
for input_file in input_files:
    print(f'Left: {len(input_files) - count}')
    count += 1
    time_series_values = np.load(f'{join(INPUT_PATH, input_file)}.npy')
    time_series_values = np.delete(time_series_values, TIME_FEATURE_INDEX, axis=0)  # delete time feature

    if iterative_values is None:
        iterative_values = initialize(time_series_values)
    else:
        iterative_values = update_array_dims(iterative_values, time_series_values)
mean, variance, sampleVariance = finalize_array_dims(iterative_values)
std = np.sqrt(variance)


print('Perform mean std standarization')
count = 0
for input_file in input_files:
    print(f'Left: {len(input_files) - count}')
    count += 1
    time_series_values = np.load(f'{join(INPUT_PATH, input_file)}.npy')
    time_series_values_aux = np.delete(time_series_values, TIME_FEATURE_INDEX, axis=0)  # delete time feature

    auxiliary_mean = np.ones((time_series_values_aux.shape[0], time_series_values_aux.shape[1])) * np.transpose(np.asarray([mean]))
    auxiliary_std = np.ones((time_series_values_aux.shape[0], time_series_values_aux.shape[1])) * np.transpose(np.asarray([std]))

    time_series_values_aux = np.divide(np.subtract(time_series_values_aux, auxiliary_mean), auxiliary_std)

    time_series_values = np.append([time_series_values[TIME_FEATURE_INDEX]], time_series_values_aux, axis=0)
    np.save(f'{join(OUTPUT_PATH, input_file)}.npy', time_series_values)
