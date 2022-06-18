import argparse
import csv
import sys
import logging
import random
import time
from os import listdir
from bisect import bisect

from app.auxiliary_files.other_methods.util_functions import load_json, save_json
from app.auxiliary_files.other_methods.visualize import plot_hist, plot_bar, compare_multiple_lines

logger = logging.getLogger(__name__)

IN_MEMORY_BYTES = 1000000000

# TODO comment -> they are situated in order, all timestamps of the same id come together

"""
filtered containers:  ['c_5433', 'c_53101', 'c_61710', 'c_39234', 'c_28768', 'c_7674', 'c_62917', 'c_65873', 'c_52574', 'c_60032']
random containers:  ['c_63367', 'c_25279', 'c_54882', 'c_17626', 'c_28804', 'c_25591', 'c_69905', 'c_25066', 'c_54511', 'c_59638']
"""

def timeit(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        logger.info(f'{method.__name__}. Elapsed time {te - ts}')
        return result

    return timed


def parse_arguments():
    def parse_bool(s: str):
        if s.casefold() in ['1', 'true', 'yes']:
            return True
        if s.casefold() in ['0', 'false', 'no']:
            return False
        raise ValueError()

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--input-folder-path',
        type=str,
        help='Path to the input folder',
        required=False
    )
    parser.add_argument(
        '--input-file-path',
        type=str,
        help='Path to the input file',
        required=False
    )
    parser.add_argument(
        '--output-folder-path',
        type=str,
        help='Output directory path for charts',
        required=True
    )
    parser.add_argument(
        '--debug',
        type=parse_bool,
        default=False,
        help='Only process first set of rows',
        required=False
    )
    parser.add_argument(
        '--verbose',
        type=parse_bool,
        default=False,
        help='Show debug logs',
        required=False
    )
    args = parser.parse_args()

    if not args.input_folder_path and not args.input_file_path:
        raise ValueError('An input folder or an input file must be specified')

    return args


class LengthsMetric:
    def __init__(self):
        self.times = {}

    def process_row(self, row: dict):
        container_id = row['container_id']
        time_stamp = int(row['time_stamp'])

        if not container_id or not time_stamp:
            return

        if container_id not in self.times:
            self.times[container_id] = {
                'start': time_stamp,
                'end': time_stamp
            }
        else:
            if time_stamp < self.times[container_id]['start']:
                self.times[container_id]['start'] = time_stamp
            if time_stamp > self.times[container_id]['end']:
                self.times[container_id]['end'] = time_stamp

    def visualize(self, output_path: str):
        plot_hist(
            False,
            f'{output_path}/container_lengths',
            'Duration of the containers',
            'time (seconds)',
            'frequency',
            x=[value['end'] - value['start'] for value in self.times.values()],
            bins=100
        )
        save_json(f'{output_path}/container_lengths', self.times)


class NumberSamplesMetric:
    def __init__(self):
        self.number_samples = {}

    def process_row(self, row: dict):
        container_id = row['container_id']
        time_stamp = int(row['time_stamp'])

        if not container_id or not time_stamp:
            return

        if container_id not in self.number_samples:
            self.number_samples[container_id] = 0
        else:
            self.number_samples[container_id] += 1

    def visualize(self, output_path: str):
        save_json(f'{output_path}/number_samples', self.number_samples)


class CPUUsageMetric:
    def __init__(self):
        self.usage = {}
        self.relative_usage_hist = [0] * 101
        self.absolute_usage_hist_intervals = list(range(-100, 3300, 25))
        self.absolute_usage_hist = [0] * len(self.absolute_usage_hist_intervals)
        self.containers_requested_cpu_usage = load_json('../output/containers_cpu_request.json')

    def process_row(self, row: dict):
        try:
            container_id = row['container_id']
            time_stamp = int(row['time_stamp'])
            cpu_util_percent = int(row['cpu_util_percent']) / 100
        except Exception:
            return

        if not container_id or not time_stamp or not cpu_util_percent or container_id not in self.containers_requested_cpu_usage:
            return

        cpu_util_real = cpu_util_percent * self.containers_requested_cpu_usage[container_id]

        #logger.info(int(row['cpu_util_percent']))
        self.relative_usage_hist[int(row['cpu_util_percent'])] += 1
        self.absolute_usage_hist[bisect(self.absolute_usage_hist_intervals, cpu_util_real)] += 1

        if container_id not in self.usage:
            self.usage[container_id] = {
                'sum_relative': cpu_util_percent,
                'count_relative': cpu_util_percent,
                'max_relative': cpu_util_percent,
                'sum_absolute': cpu_util_real,
                'count_absolute': cpu_util_real,
                'max_absolute': cpu_util_real
            }
        else:
            self.usage[container_id]['sum_relative'] += cpu_util_percent
            self.usage[container_id]['count_relative'] += 1
            self.usage[container_id]['max_relative'] = max(cpu_util_percent, self.usage[container_id]['max_relative'])
            self.usage[container_id]['sum_absolute'] += cpu_util_real
            self.usage[container_id]['count_absolute'] += 1
            self.usage[container_id]['max_absolute'] = max(cpu_util_real, self.usage[container_id]['max_absolute'])

    def visualize(self, output_path: str):
        """
        plot_bar(
            False,
            f'containers_cpu_usage_absolute_hist',
            f'Histogram of the containers CPU usage',
            'CPU usage',
            'frequency',
            x=self.absolute_usage_hist_intervals,
            height=self.absolute_usage_hist
        )

        plot_bar(
            False,
            f'containers_cpu_usage_relative_hist',
            f'Histogram of the containers CPU usage relative to the requested',
            'CPU usage',
            'frequency',
            x=[0] * 101,
            height=self.relative_usage_hist
        )
        """
        save_json(f'{output_path}/containers_cpu_usage', self.usage)
        save_json(f'{output_path}/containers_cpu_usage_relative_hist', self.relative_usage_hist)
        save_json(f'{output_path}/containers_cpu_usage_absolute_hist', self.absolute_usage_hist)


class MemoryUsageMetric:
    def __init__(self):
        self.usage = {}

    def process_row(self, row: dict):
        try:
            container_id = row['container_id']
            time_stamp = int(row['time_stamp'])
            mem_util_percent = int(row['mem_util_percent']) / 100
        except:
            return

        if not container_id or not time_stamp or not mem_util_percent:
            return

        if container_id not in self.usage:
            self.usage[container_id] = {
                'sum_relative': mem_util_percent,
                'count_relative': mem_util_percent,
                'max_relative': mem_util_percent
            }
        else:
            self.usage[container_id]['sum_relative'] += mem_util_percent
            self.usage[container_id]['count_relative'] += 1
            self.usage[container_id]['max_relative'] = max(mem_util_percent, self.usage[container_id]['max_relative'])

    def visualize(self, output_path: str):
        """
        plot_bar(
            False,
            f'containers_cpu_usage_absolute_hist',
            f'Histogram of the containers CPU usage',
            'CPU usage',
            'frequency',
            x=self.absolute_usage_hist_intervals,
            height=self.absolute_usage_hist
        )

        plot_bar(
            False,
            f'containers_cpu_usage_relative_hist',
            f'Histogram of the containers CPU usage relative to the requested',
            'CPU usage',
            'frequency',
            x=[0] * 101,
            height=self.relative_usage_hist
        )
        """
        save_json(f'{output_path}/containers_mem_usage', self.usage)


class ExtractTimeSeriesMetric:
    def __init__(self):
        self.ids = {'c_5433': [], 'c_53101': [], 'c_61710': [], 'c_39234': [], 'c_28768': [], 'c_7674': [], 'c_62917': [], 'c_65873': [], 'c_52574': [], 'c_60032': [],
                    'c_63367': [], 'c_25279': [], 'c_54882': [], 'c_17626': [], 'c_28804': [], 'c_25591': [], 'c_69905': [], 'c_25066': [], 'c_54511': [], 'c_59638': []}
        self.ids_set = set(self.ids.keys())

    def process_row(self, row: dict):
        container_id = row['container_id']
        time_stamp = row['time_stamp']

        if not container_id or not time_stamp:  # mandatory
            return

        if container_id in self.ids_set:
            self.ids[container_id].append(row)

    def visualize(self, output_path: str):
        save_json(f'{output_path}/time_series_samples', self.ids)
        """
        for container_id, container_values in self.ids:
            container_id = self.container_time_series[0]['container_id']
            time_stamp = [int(row['time_stamp']) for row in self.container_time_series]
            sorted_rows = [x for _, x in sorted(zip(time_stamp, self.container_time_series))]
            cpu_util_percent = [float(row['cpu_util_percent']) for row in sorted_rows]
            mem_util_percent = [float(row['mem_util_percent']) for row in sorted_rows]
            time_stamp = sorted(time_stamp)
    
            lines = [
                (
                    cpu_util_percent,
                    time_stamp,
                    'cpu_util_percent'
                ),
                (
                    mem_util_percent,
                    time_stamp,
                    'mem_util_percent'
                )
            ]
    
            compare_multiple_lines(
                False,
                lines,
                'usage',
                'time',
                f'Container {container_id} cpu and memory usage',
                f'{self.output_path}/container_{container_id}_usage'
            )
        """

@timeit
def process_file(input_file_path: str, metrics, debug: bool):
    header = ['container_id', 'machine_id', 'time_stamp', 'cpu_util_percent', 'mem_util_percent', 'cpi', 'mem_gps', 'mpki', 'net_in', 'net_out', 'disk_io_percent']
    #with open(input_file_path) as input_file:
    #    total_number_lines = sum(1 for _ in input_file)
    total_number_lines = 100000  # dummy
    with open(input_file_path) as input_file:
        lines = input_file.readlines(IN_MEMORY_BYTES)
        number_lines = len(lines)
        while number_lines > 0:
            logger.debug(f'Lines left: {total_number_lines}')
            csv_reader = csv.DictReader(lines, fieldnames=header)
            for index, row in enumerate(csv_reader):
                for metric in metrics:
                    metric.process_row(row)
            total_number_lines -= number_lines

            if debug:
                number_lines = 0
            else:
                lines = input_file.readlines(IN_MEMORY_BYTES)
                number_lines = len(lines)


def main(args):
    metrics = [
        # LengthsMetric(),
        # CPUUsageMetric()
        # ExtractTimeSeriesMetric(),
        # NumberSamplesMetric(),
        MemoryUsageMetric()
    ]

    if args.input_folder_path:
        files = [f for f in listdir(args.input_folder_path) if f.endswith('.csv')]
        files.sort()
        for file in files:
            path = f'{args.input_folder_path}/{file}'
            logger.info(f'Processing file {path}')
            process_file(path, metrics, args.debug)
    else:
        logger.info(f'Processing file {args.input_file_path}')
        process_file(args.input_file_path, metrics, args.debug)

    for metric in metrics:
        logger.info(f'Showing {metric.__class__.__name__} metric results')
        metric.visualize(args.output_folder_path)


if __name__ == '__main__':
    stdout_handler = logging.StreamHandler(sys.stdout)
    logging.basicConfig(
        level=eval('logging.DEBUG'),
        format='[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s',
        handlers=[stdout_handler]
    )
    args = parse_arguments()
    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)
    main(args)
