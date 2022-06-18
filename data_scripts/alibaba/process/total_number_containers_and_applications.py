import logging
import sys
import csv

from app.auxiliary_files.other_methods.util_functions import load_json, save_json

IN_MEMORY_BYTES = 1000000000
INPUT_FILE_PATH = '/data/alibaba_2018/data/container_meta.csv'
OUTPUT_FOLDER_PATH = '/data/alibaba_2018/charts'
logger = logging.getLogger(__name__)


class TotalNumbersMetric:
    def __init__(self):
        self.container_ids = set()
        self.application_ids = {}

    def process_row(self, row: dict):
        container_id = row['container_id']
        app_du = row['app_du']

        self.container_ids.add(container_id)
        if app_du in self.application_ids:
            self.application_ids[app_du].add(container_id)
        else:
            self.application_ids[app_du] = {container_id}

    def visualize(self, output_path: str):
        logger.info(f'Number containers: {len(self.container_ids)}; Number applications: {len(self.application_ids)}')
        self.application_ids = {key: list(value) for key, value in self.application_ids.items()}
        save_json(f'{output_path}/link_between_applications_and_containers', self.application_ids)



def process_file(input_file_path: str, metrics):
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
            lines = input_file.readlines(IN_MEMORY_BYTES)
            number_lines = len(lines)


def main():
    metric = TotalNumbersMetric()
    process_file(INPUT_FILE_PATH, [metric])
    metric.visualize(OUTPUT_FOLDER_PATH)


if __name__ == '__main__':
    stdout_handler = logging.StreamHandler(sys.stdout)
    logging.basicConfig(
        level=eval('logging.DEBUG'),
        format='[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s',
        handlers=[stdout_handler]
    )
    logging.basicConfig(level=logging.DEBUG)
    main()
