from os import listdir
from os.path import isfile, exists, join

import numpy as np

from app.auxiliary_files.other_methods.util_functions import load_csv

INPUT_PATH = '/home/ferran/Documents/bsc/time_series/workloads_resource_prediction/data/google_2019_a/global_data/second_preprocess_standarized'
OUTPUT_PATH = '/home/ferran/Documents/bsc/time_series/workloads_resource_prediction/data/google_2019_a/global_data/second_preprocess_standarized_normalized'
TIME_FEATURE_INDEX = 0  # wont be normalized

input_files = [
    filename.replace('.npy', '') for filename in listdir(INPUT_PATH)
    if isfile(join(INPUT_PATH, filename))
    and filename.endswith('.npy')
]

print('Check min and max for each feature')
header = load_csv(f'{INPUT_PATH}/header.csv')[0]
feature_maxs = np.zeros(len(header) - 1)
feature_mins = np.ones(len(header) - 1)
count = 0
for input_file in input_files:
    print(f'Left: {len(input_files) - count}')
    count += 1
    time_series_values = np.load(f'{join(INPUT_PATH, input_file)}.npy')
    time_series_values = np.delete(time_series_values, TIME_FEATURE_INDEX, axis=0)  # delete time feature

    max_values = np.max(time_series_values, axis=1)
    min_values = np.min(time_series_values, axis=1)

    feature_maxs = np.maximum(feature_maxs, max_values)
    feature_mins = np.minimum(feature_mins, min_values)


print('Perform min max normalization')
count = 0
for input_file in input_files:
    print(f'Left: {len(input_files) - count}')
    count += 1

    time_series_values = np.load(f'{join(INPUT_PATH, input_file)}.npy')
    time_series_values_aux = np.delete(time_series_values, TIME_FEATURE_INDEX, axis=0)  # delete time feature

    auxiliary_max = np.ones((time_series_values_aux.shape[0], time_series_values_aux.shape[1])) * np.transpose(np.asarray([feature_maxs]))
    auxiliary_min = np.ones((time_series_values_aux.shape[0], time_series_values_aux.shape[1])) * np.transpose(np.asarray([feature_mins]))

    time_series_values_aux = (time_series_values_aux - auxiliary_min) / (auxiliary_max - auxiliary_min)

    time_series_values = np.append([time_series_values[TIME_FEATURE_INDEX]], time_series_values_aux, axis=0)
    np.save(f'{join(OUTPUT_PATH, input_file)}.npy', time_series_values)
