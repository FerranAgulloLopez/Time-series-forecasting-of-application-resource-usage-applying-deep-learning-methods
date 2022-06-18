import numpy as np

from app.auxiliary_files.data_methods.time_series_preprocessing import log_transformation, sum, max_moving_average_inverse, mean_max_moving_average_inverse_strange_5
from app.auxiliary_files.other_methods.visualize import compare_multiple_lines


FEATURES_TO_SHOW = {
    'cpu_maximum_usage',
    'cpu_cycles_per_instruction',
    'memory_maximum_usage',
    'page_cache_memory',
    'memory_accesses_per_instruction',
}


def main(time_series_id, time_series_times, time_series_values, features, init_position, end_position):
    _, _, _, time_series_values_list = sum(
        {"values": [
              0.3787201,
              0.38921303,
              0.34617429,
              2.82996245,
              0.4688104,
              0.46957,
              0.48577522,
              0.7966062,
              2.2326829,
              0.34009445,
              0.34373566,
              0.34816563,
              0.35904799,
              0.46534742
            ]},
        features,
        [time_series_id],
        [time_series_times],
        [time_series_values]
    )

    _, _, _, time_series_values_list = log_transformation(
        {"min": -4},
        features,
        [time_series_id],
        [time_series_times],
        time_series_values_list
    )

    _, _, _, time_series_values_list = max_moving_average_inverse(
        {'window_size': 10},
        features,
        [time_series_id],
        [time_series_times],
        time_series_values_list
    )

    _, _, _, time_series_values_list = mean_max_moving_average_inverse_strange_5(
        {'window_size': 20},
        features,
        [time_series_id],
        [time_series_times],
        time_series_values_list
    )

    time_series = time_series_values_list[0]

    lines = []
    for feature_index, feature_label in enumerate(features):
        if feature_label in FEATURES_TO_SHOW:
            lines.append(
                (
                    time_series[feature_index],
                    list(range(init_position, end_position)),
                    feature_label
                )
            )

    compare_multiple_lines(
        False,
        lines,
        'y',
        'time',
        '',
        f'./features_{time_series_id}'
    )


if __name__ == '__main__':
    features = [
        'cpu_average_usage',
        'cpu_maximum_usage',
        'cpu_random_sampled_usage',
        'cpu_cycles_per_instruction',
        'memory_average_usage',
        'memory_maximum_usage',
        'assigned_memory',
        'page_cache_memory',
        'memory_accesses_per_instruction',
        'cpu_usage_distribution_20',
        'cpu_usage_distribution_40',
        'cpu_usage_distribution_60',
        'cpu_usage_distribution_80',
        'cpu_usage_distribution_100'
    ]
    time_series = np.load('./second_preprocess_standarized/361022904460_0.npy')
    time_series = time_series[:, 500:1000]
    time_series_times = time_series[0]
    time_series_values = np.delete(time_series, [0], axis=0)
    main('361022904460_0', time_series_times, time_series_values, features, 500, 1000)
    time_series = np.load('./second_preprocess_standarized/343677360348_3.npy')
    time_series = time_series[:, 1500:2000]
    time_series_times = time_series[0]
    time_series_values = np.delete(time_series, [0], axis=0)
    main('343677360348_3', time_series_times, time_series_values, features, 1500, 2000)
