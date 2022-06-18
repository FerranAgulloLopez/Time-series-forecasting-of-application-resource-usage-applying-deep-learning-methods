import logging
from typing import List, Dict
import csv
import os

import numpy as np

from app.auxiliary_files.other_methods.util_functions import save_csv, timeit, load_json

logger = logging.getLogger(__name__)

INPUT_FILE_PATH = './container_usage.csv'
CONTAINER_LENGTHS = '../output/number_samples.json'
FILTERED_IDS = './final_filtered_containers.json'
OUTPUT_FOLDER = '/tmp/processed'
IN_MEMORY_BYTES = 300000000
HEADER = ['time_stamp', 'cpu_util_percent', 'mem_util_percent', 'cpi', 'mem_gps', 'mpki', 'net_in', 'net_out', 'disk_io_percent']
DEBUG = True


class ExtractContainersCSV:
    def __init__(self, output_path: str, filter_ids: set, container_lengths: dict):
        self.container_time_series = None
        self.last_id = None
        self.output_path = output_path
        self.filter_ids = filter_ids
        self.time_series = {}
        self.container_lengths_filtered = {}
        for container_id in self.filter_ids:
            self.container_lengths_filtered[container_id] = container_lengths[container_id]

    def process_row(self, row: dict):
        container_id = row['container_id']
        time_stamp = row['time_stamp']

        if not container_id or not time_stamp:  # mandatory
            return
        if container_id not in self.filter_ids:
            return

        values = [row[label] if label in row and row[label] != '' else np.nan for label in HEADER]

        if container_id in self.time_series:
            self.time_series[container_id] += [values]
            if len(self.time_series[container_id]) == self.container_lengths_filtered[container_id]:
                self.__save(container_id, self.time_series[container_id])
                del self.time_series[container_id]
        else:
            self.time_series[container_id] = [values]

    def __save(self, container_id: str, values: list):
        values = np.asarray(values, dtype=float)
        values = np.transpose(values)

        if os.path.exists(f'{self.output_path}/{container_id}.npy'):
            previous_values = np.load(f'{self.output_path}/{container_id}.npy')
            values = np.concatenate((previous_values, values), axis=1)
        np.save(f'{self.output_path}/{container_id}', np.asarray(values))

    def checkpoint(self):
        for container_id, values in self.time_series.items():
            self.__save(container_id, values)
        self.time_series = {}

    def visualize(self, output_path: str):
        pass


# each job numpy array is saved in the auxiliary directory (not enough memory to load everything)
@timeit
def process_file(file_path: str, output_directory_path: str):
    filtered_ids = set(load_json(FILTERED_IDS))
    container_lengths = load_json(CONTAINER_LENGTHS)
    extractor = ExtractContainersCSV(output_directory_path, filtered_ids, container_lengths)
    del filtered_ids, container_lengths

    features = ['container_id', 'machine_id', 'time_stamp', 'cpu_util_percent', 'mem_util_percent', 'cpi', 'mem_gps', 'mpki', 'net_in', 'net_out', 'disk_io_percent']
    features_types = ['str', 'str', 'int', 'float', 'float', 'float', 'float', 'float', 'float', 'float', 'float']
    logger.info(f'features: {features}')
    logger.info(f'features types: {features_types}')
    save_csv(f'{output_directory_path}/header', [['time_stamp', 'cpu_util_percent', 'mem_util_percent', 'cpi', 'mem_gps', 'mpki', 'net_in', 'net_out', 'disk_io_percent'], ['int', 'float', 'float', 'float', 'float', 'float', 'float', 'float', 'float']])

    # create and save numpy arrays
    total_number_lines = 100000  # dummy
    with open(file_path) as input_file:
        lines = input_file.readlines(IN_MEMORY_BYTES)
        number_lines = len(lines)
        while number_lines > 0:
            logger.debug(f'Lines left: {total_number_lines}')
            csv_reader = csv.DictReader(lines, fieldnames=features)
            for index, row in enumerate(csv_reader):
                extractor.process_row(row)
            total_number_lines -= number_lines
            extractor.checkpoint()

            if DEBUG:
                number_lines = 0
            else:
                lines = input_file.readlines(IN_MEMORY_BYTES)
                number_lines = len(lines)
    extractor.visualize(output_directory_path)


def main():
    logger.info(f'Processing file {INPUT_FILE_PATH}')
    process_file(INPUT_FILE_PATH, OUTPUT_FOLDER)


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    main()
