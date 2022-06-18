import argparse
import csv
import sys
import logging
import random
import time
from os import listdir

from app.auxiliary_files.other_methods.util_functions import load_json, save_json
from app.auxiliary_files.other_methods.visualize import plot_hist, plot_bar, compare_multiple_lines

logger = logging.getLogger(__name__)

IN_MEMORY_BYTES = 1000000000

# TODO comment -> they are situated in order, all timestamps of the same id come together


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


class TotalNumbersMetric:
    def __init__(self):
        self.container_ids = set()
        self.application_ids = set()

    def process_row(self, row: dict):
        container_id = row['container_id']
        app_du = row['app_du']

        self.container_ids.add(container_id)
        self.application_ids.add(app_du)

    def visualize(self, output_path: str):
        logger.info(f'Number containers: {len(self.container_ids)}; Number applications: {len(self.application_ids)}')
        save_json(f'{output_path}/number_container_and_applications', {
            'number_containers': len(self.container_ids),
            'number_applications': len(self.application_ids)
        })


class RequestedUsageMetric:
    def __init__(self):
        self.containers = {}

    def process_row(self, row: dict):
        container_id = row['container_id']
        cpu_request = int(row['cpu_request'])

        if container_id in self.containers:
            if self.containers[container_id] != cpu_request:
                raise Exception('Meh')
        else:
            self.containers[container_id] = cpu_request

    def visualize(self, output_path: str):
        save_json(f'{output_path}/containers_cpu_request', self.containers)


@timeit
def process_file(input_file_path: str, metrics, debug: bool):
    header = ['container_id', 'machine_id', 'time_stamp', 'app_du', 'status', 'cpu_request', 'cpu_limit', 'mem_size']
    with open(input_file_path) as input_file:
        total_number_lines = sum(1 for _ in input_file)
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
        TotalNumbersMetric(),
        RequestedUsageMetric()
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
