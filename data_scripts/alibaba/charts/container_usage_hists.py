import numpy as np
import random

from app.auxiliary_files.other_methods.visualize import plot_bar, plot_hist, compare_multiple_lines, compare_multiple_lines_matrix
from app.auxiliary_files.other_methods.util_functions import load_json, save_json


def main():
    containers_mem_usage = load_json('containers_mem_usage.json')
    containers_average_mem_usage = [(values['sum_relative'] / values['count_relative']) for key, values in containers_mem_usage.items()]
    containers_max_mem_usage = [values['max_relative'] for key, values in containers_mem_usage.items()]

    plot_hist(
        False,
        f'average_memory_usage',
        '',
        'average memory usage',
        'frequency',
        x=containers_average_mem_usage,
        bins=100
    )

    plot_hist(
        False,
        f'maximum_memory_usage',
        '',
        'maximum memory usage',
        'frequency',
        x=containers_max_mem_usage,
        bins=100
    )
    """

    charts_cpu = []
    time_series_samples = load_json('time_series_samples.json')
    # containers_ids = random.sample(list(time_series_samples.keys()), 5)
    containers_ids = ['c_25591', 'c_5433', 'c_28768', 'c_7674', 'c_54511']
    for container_id in containers_ids:
        container_values = time_series_samples[container_id]
        time_stamp = [int(row['time_stamp']) for row in container_values]
        sorted_rows = [x for _, x in sorted(zip(time_stamp, container_values))]
        cpu_util_percent = [float(row['cpu_util_percent']) for row in sorted_rows]
        time_stamp = sorted(time_stamp)

        charts_cpu.append((
            [
                (
                    cpu_util_percent[8500:20500],
                    time_stamp[8500:20500],
                    ''
                )
            ],
            'cpu usage',
            ''
        ))

    compare_multiple_lines_matrix(
        False,
        charts_cpu,
        '',
        'time',
        f'time_series_samples/cpu_time_series_samples_simple',
        ncols=1,
        ylim=(0, 40),
        legend=False
    )
    """
    containers_cpu_usage = load_json('containers_cpu_usage.json')
    containers_average_averaged_cpu_usage_absolute = [(values['sum_absolute'] / values['count_absolute']) / 100 for key, values in containers_cpu_usage.items()]
    containers_average_averaged_cpu_usage_relative = [values['sum_relative'] / values['count_relative'] for key, values in containers_cpu_usage.items()]
    containers_max_averaged_cpu_usage_absolute = [values['max_absolute'] / 100 for key, values in containers_cpu_usage.items()]
    containers_max_averaged_cpu_usage_relative = [values['max_relative'] for key, values in containers_cpu_usage.items()]

    plot_hist(
        False,
        f'average_averaged_cpu_usage_absolute',
        '',
        'average CPU usage',
        'frequency',
        x=containers_average_averaged_cpu_usage_absolute,
        bins=100
    )

    plot_hist(
        False,
        f'average_averaged_cpu_usage_relative',
        '',
        'average CPU usage percentage',
        'frequency',
        x=containers_average_averaged_cpu_usage_relative,
        bins=100
    )

    plot_hist(
        False,
        f'max_average_cpu_usage_absolute',
        '',
        'maximum averaged CPU usage',
        'frequency',
        x=containers_max_averaged_cpu_usage_absolute,
        bins=100
    )

    plot_hist(
        False,
        f'max_average_cpu_usage_relative',
        '',
        'maximum averaged CPU usage percentage',
        'frequency',
        x=containers_max_averaged_cpu_usage_relative,
        bins=100
    )

    containers_cpu_usage = load_json('containers_cpu_usage.json')
    containers_requested_usage = load_json('containers_cpu_request.json')
    containers_difference_average_requested_absolute = [(containers_requested_usage[key] - (values['sum_absolute'] / values['count_absolute'])) / 100 for key, values in containers_cpu_usage.items()]
    containers_difference_average_max_absolute = [(values['max_absolute'] - (values['sum_absolute'] / values['count_absolute'])) / 100 for key, values in containers_cpu_usage.items()]

    plot_hist(
        False,
        f'difference_average_requested_cpu_usage_absolute',
        '',
        'CPU usage difference',
        'frequency',
        x=containers_difference_average_requested_absolute,
        bins=100
    )

    plot_hist(
        False,
        f'difference_average_max_cpu_usage_absolute',
        '',
        'CPU usage',
        'frequency',
        x=containers_difference_average_max_absolute,
        bins=100
    )

    containers_length = load_json('container_lengths.json')
    containers_cpu_usage = load_json('containers_cpu_usage.json')
    containers_requested_usage = load_json('containers_cpu_request.json')
    filtered_containers = [key
                          for key, values in containers_cpu_usage.items()
                           if values['max_relative'] > 0.25 and 100 < (values['max_absolute'] - (values['sum_absolute'] / values['count_absolute']))
                           and (containers_length[key]['end'] - containers_length[key]['start']) >= 691190 and containers_requested_usage[key] == 400]
    print(f'First filtered containers count: {len(filtered_containers)}')
    container_applications = load_json('link_between_containers_and_applications.json')
    filtered_containers_applications = {}
    for container in filtered_containers:
        application = container_applications[container]
        if application in filtered_containers_applications:
            filtered_containers_applications[application] += [container]
        else:
            filtered_containers_applications[application] = [container]
    final_filtered_set = [random.sample(list(values), 1)[0] for _, values in filtered_containers_applications.items()]
    print(f'Final filtered containers count: {len(final_filtered_set)}')
    random.shuffle(filtered_containers)
    save_json('final_filtered_containers', final_filtered_set)

    container_times = load_json('container_lengths.json')
    plot_hist(
        False,
        'container_lengths',
        '',
        'time (seconds)',
        'frequency',
        x=[value['end'] - value['start'] for value in container_times.values()],
        bins=100
    )

    """
    charts_cpu = []
    charts_memory = []
    time_series_samples = load_json('time_series_samples.json')
    containers_ids = ['c_5433', 'c_53101', 'c_61710', 'c_39234', 'c_28768', 'c_7674', 'c_62917', 'c_65873', 'c_52574', 'c_60032']
    for container_id in containers_ids:
        container_values = time_series_samples[container_id]
        time_stamp = [int(row['time_stamp']) for row in container_values]
        sorted_rows = [x for _, x in sorted(zip(time_stamp, container_values))]
        cpu_util_percent = [float(row['cpu_util_percent']) for row in sorted_rows]
        mem_util_percent = [float(row['mem_util_percent']) for row in sorted_rows]
        time_stamp = sorted(time_stamp)

        charts_cpu.append((
            [
                (
                    cpu_util_percent,
                    time_stamp,
                    'cpu_util_percent'
                )
            ],
            container_id,
            f'CPU usage of 10 randomly selected time series from the filtered ones'
        ))

        charts_memory.append((
            [
                (
                    mem_util_percent,
                    time_stamp,
                    'mem_util_percent'
                )
            ],
            container_id,
            f'Memory usage of 10 randomly selected time series from the filtered ones'
        ))

    compare_multiple_lines_matrix(
        False,
        charts_cpu,
        '',
        'time',
        f'time_series_samples/cpu_time_series_samples_filtered',
        ncols=1
    )

    compare_multiple_lines_matrix(
        False,
        charts_memory,
        '',
        'time',
        f'time_series_samples/memory_time_series_samples_filtered',
        ncols=1
    )
    """


    def mean_moving_average_inverse(array, window_size=5):
        for time_position in range(array.shape[0] - 1, window_size, -1):
            array[time_position] = np.mean(array[(time_position - window_size + 1):(time_position + 1)])
        return array

    def reduction(array):
        new_array = np.zeros(array.shape[0] // 2)
        index = 0
        for time_position in range(0, array.shape[0] - 1, 2):
            new_array[index] = array[time_position]
            index += 1
        return new_array

    charts_cpu = []
    charts_memory = []
    time_series_samples = load_json('time_series_samples.json')
    containers_ids = ['c_63367', 'c_25279', 'c_54882', 'c_17626', 'c_28804']  # , 'c_25591', 'c_69905', 'c_25066', 'c_54511', 'c_59638']
    for container_id in containers_ids:
        container_values = time_series_samples[container_id]
        time_stamp = [int(row['time_stamp']) for row in container_values]
        sorted_rows = [x for _, x in sorted(zip(time_stamp, container_values))]
        cpu_util_percent = [float(row['cpu_util_percent']) for row in sorted_rows]
        cpu_util_percent = reduction(reduction(mean_moving_average_inverse(np.asarray(cpu_util_percent))))
        mem_util_percent = [float(row['mem_util_percent']) for row in sorted_rows]
        time_stamp = sorted(time_stamp)

        charts_cpu.append((
            [
                (
                    cpu_util_percent,
                    reduction(reduction(np.asarray(time_stamp))),
                    'cpu_average_usage_relative'
                )
            ],
            container_id,
            f''
        ))

        charts_memory.append((
            [
                (
                    mem_util_percent,
                    time_stamp,
                    'memory_average_usage_relative'
                )
            ],
            container_id,
            f''
        ))

    compare_multiple_lines_matrix(
        False,
        charts_cpu,
        '',
        'time',
        f'time_series_samples/cpu_time_series_samples',
        ncols=1
    )
    """
    """
    compare_multiple_lines_matrix(
        False,
        charts_memory,
        '',
        'time',
        f'time_series_samples/memory_time_series_samples',
        ncols=1
    )



if __name__ == '__main__':
    main()
