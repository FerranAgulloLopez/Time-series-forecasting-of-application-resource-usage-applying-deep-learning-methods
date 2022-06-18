import numpy as np
from app.auxiliary_files.other_methods.visualize import compare_multiple_lines


def mean_moving_average_inverse(time_series_values: np.ndarray, window_size) -> np.ndarray:
    for time_position in range(time_series_values.shape[0] - 1, window_size, -1):
        time_series_values[time_position] = np.mean(
            time_series_values[(time_position - window_size + 1):(time_position + 1)]
        )
    return time_series_values


def mean_moving_average_pre(time_series_values: np.ndarray, window_size) -> np.ndarray:
    for time_position in range(time_series_values.shape[0] - window_size):  # TODO only works with one feature arrays
        time_series_values[time_position] = np.mean(
            time_series_values[time_position:(time_position + window_size)]
        )
    return time_series_values


def max_moving_average_inverse(time_series_values: np.ndarray, window_size) -> np.ndarray:
    for time_position in range(time_series_values.shape[0] - 1, window_size, -1):
        time_series_values[time_position] = np.max(
            time_series_values[(time_position - window_size + 1):(time_position + 1)]
        )
    return time_series_values


def last_maximum_mean_moving_average_inverse(time_series_values: np.ndarray, window_size) -> np.ndarray:
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
    return time_series_values


def main():
    time_series = np.load('./third_preprocess_compressed/c_41134.npy')
    time_series = time_series[1, 700:1000]  # 1250:1500

    lines = [
        (
            time_series,
            list(range(0, time_series.shape[0])),
            'raw'
        ),
        (
            mean_moving_average_pre(time_series.copy(), 5),
            list(range(0, time_series.shape[0])),
            'window 5'
        ),
        (
            mean_moving_average_pre(time_series.copy(), 15),
            list(range(0, time_series.shape[0])),
            'window 15'
        ),
        (
            mean_moving_average_pre(time_series.copy(), 30),
            list(range(0, time_series.shape[0])),
            'window 30'
        )
    ]
    compare_multiple_lines(
        False,
        lines,
        'y',
        'time',
        '',
        f'./future_mean_filter_example',
        ylim=(1.2, 2.2),
        linewidth=5
    )

    lines = [
        (
            time_series,
            list(range(0, time_series.shape[0])),
            'raw'
        ),
        (
            mean_moving_average_inverse(time_series.copy(), 5),
            list(range(0, time_series.shape[0])),
            'window 5'
        ),
        (
            mean_moving_average_inverse(time_series.copy(), 15),
            list(range(0, time_series.shape[0])),
            'window 15'
        ),
        (
            mean_moving_average_inverse(time_series.copy(), 30),
            list(range(0, time_series.shape[0])),
            'window 30'
        )
    ]
    compare_multiple_lines(
        False,
        lines,
        'y',
        'time',
        '',
        f'./past_mean_filter_example',
        ylim=(1.2, 2.2),
        linewidth=5
    )

    lines = [
        (
            time_series,
            list(range(0, time_series.shape[0])),
            'raw'
        ),
        (
            max_moving_average_inverse(time_series.copy(), 5),
            list(range(0, time_series.shape[0])),
            'window 5'
        ),
        (
            max_moving_average_inverse(time_series.copy(), 15),
            list(range(0, time_series.shape[0])),
            'window 15'
        ),
        (
            max_moving_average_inverse(time_series.copy(), 30),
            list(range(0, time_series.shape[0])),
            'window 30'
        )
    ]
    compare_multiple_lines(
        False,
        lines,
        'y',
        'time',
        '',
        f'./past_max_filter_example',
        ylim=(1.2, 2.2),
        linewidth=5
    )

    lines = [
        (
            time_series,
            list(range(0, time_series.shape[0])),
            'raw'
        ),
        (
            last_maximum_mean_moving_average_inverse(time_series.copy(), 5),
            list(range(0, time_series.shape[0])),
            'window 5'
        ),
        (
            last_maximum_mean_moving_average_inverse(time_series.copy(), 15),
            list(range(0, time_series.shape[0])),
            'window 15'
        ),
        (
            last_maximum_mean_moving_average_inverse(time_series.copy(), 30),
            list(range(0, time_series.shape[0])),
            'window 30'
        )
    ]
    compare_multiple_lines(
        False,
        lines,
        'y',
        'time',
        '',
        f'./last_maximum_mean_filter_example',
        ylim=(1.2, 2.2),
        linewidth=5
    )

    lines = [
        (
            time_series,
            list(range(0, time_series.shape[0])),
            'raw'
        ),
        (
            mean_moving_average_pre(max_moving_average_inverse(time_series.copy(), 5), 5),
            list(range(0, time_series.shape[0])),
            'window 5'
        ),
        (
            mean_moving_average_pre(max_moving_average_inverse(time_series.copy(), 15), 15),
            list(range(0, time_series.shape[0])),
            'window 15'
        ),
        (
            mean_moving_average_pre(max_moving_average_inverse(time_series.copy(), 30), 30),
            list(range(0, time_series.shape[0])),
            'window 30'
        )
    ]
    compare_multiple_lines(
        False,
        lines,
        'y',
        'time',
        '',
        f'./combination_past_max_&_future_mean_example',
        ylim=(1.2, 2.2),
        linewidth=5
    )

    lines = [
        (
            time_series,
            list(range(0, time_series.shape[0])),
            'raw'
        ),
        (
            last_maximum_mean_moving_average_inverse(max_moving_average_inverse(time_series.copy(), 5), 5),
            list(range(0, time_series.shape[0])),
            'window 5'
        ),
        (
            last_maximum_mean_moving_average_inverse(max_moving_average_inverse(time_series.copy(), 15), 15),
            list(range(0, time_series.shape[0])),
            'window 15'
        ),
        (
            last_maximum_mean_moving_average_inverse(max_moving_average_inverse(time_series.copy(), 30), 30),
            list(range(0, time_series.shape[0])),
            'window 30'
        )
    ]
    compare_multiple_lines(
        False,
        lines,
        'y',
        'time',
        '',
        f'./combination_past_max_&_last_maximum_mean_filter_example',
        ylim=(1.2, 2.2),
        linewidth=5
    )


if __name__ == '__main__':
    main()
