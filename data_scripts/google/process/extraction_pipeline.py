import gzip
import shutil
import argparse
import csv
import json
import logging
import os
import time
from os import listdir
import numpy as np
from numpy import VisibleDeprecationWarning

IN_MEMORY_BYTES = 1000000000
logger = logging.getLogger(__name__)


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
        help='Path to the output file',
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


@timeit
def extract_file(input_file_path: str, output_file_path: str, remove: bool):
    with gzip.open(input_file_path, 'rb') as f_in:
        with open(output_file_path, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
    if remove:
        os.remove(input_file_path)


@timeit
class ToNumpyArrays:
    def __init__(self, output_folder_path: str):
        self.header = [
            'time_stamp',
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
        self.header_types = [
            'int',
            'float',
            'float',
            'float',
            'float',
            'float',
            'float',
            'float',
            'float',
            'float',
            'float',
            'float',
            'float',
            'float',
            'float'
        ]
        with open(f'{output_folder_path}/header.csv', 'w') as output_file:
            csv_writer = csv.writer(output_file)
            csv_writer.writerow(self.header)
            csv_writer.writerow(self.header_types)

        self.output_folder_path = output_folder_path
        self.last_id = None
        self.time_series_values = []

    def __process_row(self, obj: dict):
        _id = obj['id']
        values = [obj[key] if key in obj else np.nan for key in self.header[:-5]]
        if len(obj['cpu_usage_distribution']) != 11:
            values += [np.nan] * 5
        else:
            values += [
                obj['cpu_usage_distribution'][2],
                obj['cpu_usage_distribution'][4],
                obj['cpu_usage_distribution'][6],
                obj['cpu_usage_distribution'][8],
                obj['cpu_usage_distribution'][10]
            ]
        if self.last_id is None:
            self.last_id = _id
            self.time_series_values = [values]
        else:
            if self.last_id == _id:
                self.time_series_values.append(values)
            else:
                self.__save_last()
                self.last_id = _id
                self.time_series_values = [values]

    def __save_last(self):
        self.time_series_values = np.asarray(self.time_series_values, dtype=float)
        self.time_series_values = np.transpose(self.time_series_values)
        np.save(f'{self.output_folder_path}/{self.last_id}', self.time_series_values)

    def process(self, input_file_path: str):
        with open(input_file_path) as input_file:
            total_number_lines = sum(1 for _ in input_file)
        with open(input_file_path) as input_file:
            lines = input_file.readlines(IN_MEMORY_BYTES)
            number_lines = len(lines)
            while number_lines > 0:
                logger.debug(f'Lines left: {total_number_lines}')
                for line in lines:
                    obj = json.loads(line)
                    self.__process_row(obj)
                total_number_lines -= number_lines
                lines = input_file.readlines(IN_MEMORY_BYTES)
                number_lines = len(lines)


@timeit
def process_file(input_file_path: str, to_numpy_arrays: ToNumpyArrays):
    new_input_file_path = f'{input_file_path.replace(".gzip", "")}.csv'
    logger.info('Extracting')
    extract_file(input_file_path, new_input_file_path, remove=False)
    logger.info('Converting to numpy')
    to_numpy_arrays.process(new_input_file_path)
    os.remove(new_input_file_path)


def main(args):
    to_numpy_arrays = ToNumpyArrays(args.output_folder_path)
    if args.input_folder_path:
        files = listdir(args.input_folder_path)
        files.sort()
        for file in files:
            path = f'{args.input_folder_path}/{file}'
            logger.info(f'Processing file {path}')
            process_file(path, to_numpy_arrays)
    else:
        logging.info(f'Processing file {args.input_file_path}')
        process_file(args.input_file_path, to_numpy_arrays)


if __name__ == '__main__':
    args = parse_arguments()
    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)
    main(args)
