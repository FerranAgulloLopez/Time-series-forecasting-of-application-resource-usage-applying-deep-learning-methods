from os import listdir
from os.path import isfile, exists, join

import numpy as np

from app.auxiliary_files.other_methods.util_functions import load_csv

INPUT_PATH = '/home/ferran/Documents/bsc/time_series/workloads_resource_prediction/data/alibaba_2018/data/first_preprocess'
OUTPUT_PATH = '/home/ferran/Documents/bsc/time_series/workloads_resource_prediction/data/alibaba_2018/data/first_preprocess_compressed'
TIME_FEATURE_INDEX = 0
WINDOW_TIME_SIZE = 300


input_files = [
    filename.replace('.npy', '') for filename in listdir(INPUT_PATH)
    if isfile(join(INPUT_PATH, filename))
    and filename.endswith('.npy')
]

count = 0
for input_file in input_files:
    print(f'Left: {len(input_files) - count}')
    count += 1
    time_series_values = np.load(f'{join(INPUT_PATH, input_file)}.npy')
    time_series_times = time_series_values[TIME_FEATURE_INDEX, :]

    time_series_values_new = None
    init_time_index = 0
    for current_time_index in range(1, time_series_times.shape[0]):
        if time_series_times[current_time_index] >= (time_series_times[init_time_index] + WINDOW_TIME_SIZE):
            window_values = time_series_values[:, init_time_index:current_time_index]
            mean_values = np.mean(window_values, axis=1)
            max_values = np.max(window_values, axis=1)
            percentiles_values = np.percentile(window_values, [20, 40, 60, 80], axis=1)

            window_values = [
                time_series_times[current_time_index - 1],  # time
                mean_values[1],  # average cpu
                max_values[1],  # max cpu
                percentiles_values[0, 1],  # cpu percentile 20
                percentiles_values[1, 1],  # cpu percentile 40
                percentiles_values[2, 1],   # cpu percentile 60
                percentiles_values[3, 1],   # cpu percentile 80
                mean_values[2],  # average memory
                max_values[2],  # max memory
                percentiles_values[0, 2],   # memory percentile 20
                percentiles_values[1, 2],  # memory percentile 40
                percentiles_values[2, 2],  # memory percentile 60
                percentiles_values[3, 2],  # memory percentile 80
                mean_values[3],  # average network_in
                max_values[3],  # max network_in
                mean_values[4],  # average network_out
                max_values[4],  # max network_out
                mean_values[5],  # average disk_usage
                max_values[5]  # max disk_usage
            ]
            window_values = np.expand_dims(window_values, axis=1)

            if time_series_values_new is None:
                time_series_values_new = window_values
            else:
                time_series_values_new = np.concatenate((time_series_values_new, window_values), axis=1)
            init_time_index = current_time_index

    np.save(f'{join(OUTPUT_PATH, input_file)}.npy', time_series_values_new)
