import math
import numpy as np
import random
import matplotlib.pyplot as plt
from os import listdir
from bisect import bisect
from os.path import isfile, join
import itertools
from scipy.stats import boxcox

from app.auxiliary_files.other_methods.visualize import compare_multiple_lines


INPUT_TIME_SERIES = f'../data/google_2019_a/global_data/second_preprocess_standarized_normalized'
COUNT = 5
FEATURE_TO_PREDICT_INDEX = 2
OUTPUT_PATH = '/home/ferran/Documents/bsc/time_series/workloads_resource_prediction/output'


def main(input_files):
    for input_filename, input_file_path in input_files:
        time_series_id = input_filename
        time_series_values = np.load(input_file_path)[FEATURE_TO_PREDICT_INDEX]  # only target feature

        # time_series_values_transformed = np.log(time_series_values + 0.000001)
        std = np.std(time_series_values)
        mean = np.mean(time_series_values)
        for time_position in range(1, time_series_values.shape[0]):
            value = time_series_values[time_position]
            difference = value - mean
            if value == 0 or (difference < 0 and np.abs(difference) > 2*std):
                time_series_values[time_position] = time_series_values[time_position - 1]
        time_series_values_transformed = np.log(time_series_values)

        compare_multiple_lines(
            False,
            [
                (
                    time_series_values,
                    np.arange(time_series_values.shape[0]),
                    'raw'
                )
            ],
            'cpu',
            'time',
            'Log transform',
            f'{OUTPUT_PATH}/log_transform_{time_series_id}_raw'
        )
        compare_multiple_lines(
            False,
            [
                (
                    time_series_values_transformed,
                    np.arange(time_series_values_transformed.shape[0]),
                    'transformed'
                )
            ],
            'cpu',
            'time',
            'Log transform',
            f'{OUTPUT_PATH}/log_transform_{time_series_id}_transformed'
        )


if __name__ == '__main__':
    input_files = [
        (filename.replace('.npy', ''), join(INPUT_TIME_SERIES, filename)) for filename in listdir(INPUT_TIME_SERIES)
        if isfile(join(INPUT_TIME_SERIES, filename))
           and filename.endswith('.npy')
    ]

    random.shuffle(input_files)
    input_files = input_files[:COUNT]

    main(input_files)
