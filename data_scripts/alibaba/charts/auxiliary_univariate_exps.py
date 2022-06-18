import numpy as np

from app.auxiliary_files.data_methods.time_series_preprocessing import log_transformation, sum, max_moving_average_inverse, mean_max_moving_average_inverse_strange_5
from app.auxiliary_files.other_methods.visualize import compare_multiple_lines


FEATURES_TO_SHOW = {
    'average_cpu_average_usage_relative',
    'maximum_cpu_average_usage_relative',
    'average_disk_average_usage_relative',
    'maximum_disk_average_usage_relative'
}


def main(time_series_id, time_series_times, time_series_values, features, init_position, end_position):
    _, _, _, time_series_values_list = sum(
        {"values": [0.8293231410685978, 0.9069997126030365, 0.8777236007469805, 0.7067615631752456, 0.7360793569458134, 0.7849484188206659, 2.858544167686376, 2.8397778636204842, 2.919976251757168, 2.879644460096365, 2.8391935283060676, 2.8385477189535084, 0.5219586660230582, 0.5223685403149223, 0.5422420611292673, 0.5423664695192236, 1.4252378830568786, 1.3246437619650067]},
        features,
        [time_series_id],
        [time_series_times],
        [time_series_values]
    )

    _, _, _, time_series_values_list = log_transformation(
        {},
        features,
        [time_series_id],
        [time_series_times],
        [time_series_values]
    )

    _, _, _, time_series_values_list = max_moving_average_inverse(
        {'window_size': 10},
        features,
        [time_series_id],
        [time_series_times],
        [time_series_values]
    )

    _, _, _, time_series_values_list = mean_max_moving_average_inverse_strange_5(
        {'window_size': 20},
        features,
        [time_series_id],
        [time_series_times],
        [time_series_values]
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
        'average_cpu_average_usage_relative',
        'maximum_cpu_average_usage_relative',
        'percentile_20_cpu_average_usage_relative',
        'percentile_40_cpu_average_usage_relative',
        'percentile_60_cpu_average_usage_relative',
        'percentile_80_cpu_average_usage_relative',
        'average_memory_average_usage_relative',
        'maximum_memory_average_usage_relative',
        'percentile_20_memory_average_usage_relative',
        'percentile_40_memory_average_usage_relative',
        'percentile_60_memory_average_usage_relative',
        'percentile_80_memory_average_usage_relative',
        'average_network_in',
        'maximum_network_in',
        'average_network_out',
        'maximum_network_out',
        'average_disk_average_usage_relative',
        'maximum_disk_average_usage_relative'
    ]
    time_series = np.load('./third_preprocess_compressed/c_55114.npy')
    time_series = time_series[:, 500:1000]
    time_series_times = time_series[0]
    time_series_values = np.delete(time_series, [0], axis=0)
    main('c_55114', time_series_times, time_series_values, features, 500, 1000)
    time_series = np.load('./third_preprocess_compressed/c_66306.npy')
    time_series = time_series[:, 1000:1500]
    time_series_times = time_series[0]
    time_series_values = np.delete(time_series, [0], axis=0)
    main('c_66306', time_series_times, time_series_values, features, 1000, 1500)
