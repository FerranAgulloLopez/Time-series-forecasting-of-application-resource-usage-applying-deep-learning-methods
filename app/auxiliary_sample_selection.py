import math
import numpy as np
import random
import matplotlib.pyplot as plt
from os import listdir
from bisect import bisect
from os.path import isfile, join
import itertools

from app.auxiliary_files.other_methods.visualize import compare_multiple_lines_points_color_extended


INPUT_TIME_SERIES = f'../data/google_2019_a/global_data/second_preprocess_standarized'
COUNT = 5
FEATURE_TO_PREDICT_INDEX = 2
OUTPUT_PATH = '/home/ferran/Documents/bsc/time_series/workloads_resource_prediction/output'


def show_intervals(input_files):
    WINDOW_SIZE = 5

    for input_filename, input_file_path in input_files:
        time_series_id = input_filename
        time_series_values = np.load(input_file_path)[FEATURE_TO_PREDICT_INDEX]  # only target feature

        time_series_std = np.std(time_series_values)
        difference_intervals = [time_series_std * value for value in [0, 0.5, 2]]
        difference_intervals_count = [0] * len(difference_intervals)
        difference_intervals_positions = [[] for _ in range(len(difference_intervals))]
        for time_position in range(WINDOW_SIZE, time_series_values.shape[0]):
            lag_mean = np.mean(time_series_values[(time_position - WINDOW_SIZE):time_position])
            difference = time_series_values[time_position] - lag_mean

            interval_index = bisect(difference_intervals, difference)
            if interval_index >= len(difference_intervals):
                interval_index = len(difference_intervals) - 1
            difference_intervals_count[interval_index] += 1
            difference_intervals_positions[interval_index].append((time_series_values[time_position], time_position))

        print(difference_intervals_count)

        for interval_index in range(len(difference_intervals)):
            compare_multiple_lines_points_color_extended(
                False,
                [
                    (
                        time_series_values,
                        np.arange(time_series_values.shape[0]),
                        'blue',
                        3,
                        'real usage'
                    )
                ],
                [(value, time_position, "red") for (value, time_position) in difference_intervals_positions[interval_index]],
                'y',
                'time',
                '',
                f'{OUTPUT_PATH}/samples_deviations_{time_series_id}_interval_{interval_index}'
            )


def show_balanced_selection(input_files):
    WINDOW_SIZE = 5

    for input_filename, input_file_path in input_files:
        time_series_id = input_filename
        time_series_values = np.load(input_file_path)[FEATURE_TO_PREDICT_INDEX]  # only target feature

        time_series_std = np.std(time_series_values)
        difference_intervals = [time_series_std * value for value in [0, 0.5, 2]]
        difference_intervals_count = [0] * len(difference_intervals)
        difference_intervals_positions = [[] for _ in range(len(difference_intervals))]
        for time_position in range(WINDOW_SIZE, time_series_values.shape[0]):
            lag_mean = np.mean(time_series_values[(time_position - WINDOW_SIZE):time_position])
            difference = time_series_values[time_position] - lag_mean

            interval_index = bisect(difference_intervals, difference)
            if interval_index >= len(difference_intervals):
                interval_index = len(difference_intervals) - 1
            difference_intervals_count[interval_index] += 1
            difference_intervals_positions[interval_index].append((time_series_values[time_position], time_position))

        total_samples = len(time_series_values) - WINDOW_SIZE  # caution, only get values over lag_size
        total_samples //= 2

        samples_for_interval = total_samples // len(difference_intervals)
        for interval_index in range(len(difference_intervals)):
            if len(difference_intervals_positions[interval_index]) < samples_for_interval:
                difference_intervals_positions[interval_index] = [random.choice(difference_intervals_positions[interval_index]) for _ in range(samples_for_interval)]
            else:
                difference_intervals_positions[interval_index] = random.sample(difference_intervals_positions[interval_index], samples_for_interval)

        print(f'initial selection: {difference_intervals_count};final selection: {[len(elements) for elements in difference_intervals_positions]}')

        all_points = list(itertools.chain.from_iterable(difference_intervals_positions))
        compare_multiple_lines_points_color_extended(
            False,
            [
                (
                    time_series_values,
                    np.arange(time_series_values.shape[0]),
                    'blue',
                    3,
                    'values'
                )
            ],
            [(value, time_position, "red") for (value, time_position) in all_points],
            'cpu',
            'time',
            '',
            f'{OUTPUT_PATH}/samples_deviations_{time_series_id}_final'
        )


if __name__ == '__main__':
    input_files = [
        (filename.replace('.npy', ''), join(INPUT_TIME_SERIES, filename)) for filename in listdir(INPUT_TIME_SERIES)
        if isfile(join(INPUT_TIME_SERIES, filename))
           and filename.endswith('.npy')
    ]

    random.shuffle(input_files)
    input_files = input_files[:COUNT]

    show_intervals(input_files)
    show_balanced_selection(input_files)
