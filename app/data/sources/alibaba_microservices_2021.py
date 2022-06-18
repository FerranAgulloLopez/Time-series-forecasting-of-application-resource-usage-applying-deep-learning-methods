import csv
import logging
from typing import List

import numpy as np

from app.auxiliary_files.other_methods.visualize import plot_hist, compare_multiple_lines
from app.data.data_source_abstract import DataSourceAbstract

logger = logging.getLogger(__name__)


class DataSourceAlibabaMicroservices2021(DataSourceAbstract):

    def __init__(self, config: dict, device: str):
        super().__init__(config, device)
        self.ms_resources_directory_path = f'{config["directory_path"]}/MSResource'

    # ---> Main methods

    def load(self) -> (List[np.ndarray], List[str]):
        pass

    def visualize(self, config: dict, output_path: str):
        ms_map = {}
        ms_instance_map = {}

        count = 0

        ms_follow = {
            'b7c72dccdd6db4fb7c2cecfc47215b93c30f927d59baa33785a358853fba2b46',
            '8182d457578bb18d3f24346c45921fbd642cb9bbcc5d1407e476e02a35b5578b',
            '7b973c9a30cb5fa6024d07de011805cd5333d0dd2b92f8c18d279502406355d1',
            '7f5f0de79c4e994623c8b90b4d867414cd3951eb7553d6122667b41fe394752c',
            '5b899f47919bee478321ecdd4ada8eedf49db00c23634aa800e077ac264f474c',
            'bec737fe68cdb7392bf0dae35010df7621af0c200d834ebc2398404b8d6b37b5',
            '449cf6fb36ffe9b099490765afdd19e657b2de013a3bdecb6bdb023734757e46',
            '5e79453d4a01b1bd2ccf59e3ef49ad1722a13a5622876611fe7539393a70d491',
            '93f9fc4fcdba3500ce8b9ebe983b7b87bfa9863e20813d6dbc783a8f6711e925',
            '83c59565a5b43e8a0e9a2c0faa1f007c4d5404e5239d3d5d13e17b101eff3e81',
            '73dc0b88853241955161056b003518112b11b6648ed1fedeb0f39567a0e29346',
            '9e365a75fea1a8b67440fda7e3ce24164b498ac070a13ba201e16511962dc54e',
            'd21bd8de14e8d34dbe4140736321a7a1349a198230e863e9e2acb02f8905a28d',
            '73f6d8a8dceebd7475a100eab899082be5f8ad2234356908f32e8aa84b71eb55',
            'efcee98ea21c08e758a5da71cc7184939bf9a199624c841816895e9ba90b4f44',
            'd0c2a815bb0667c75be62eb9f877e3c1365d7f14ae9cabfe634d0094b106f1b1',
            '27d4ba78d387f84e31c14d068d89994dd7cea0805d24ba7b80b2f0d226fe1906',
            'cd15e277536a82697e10dd402532ebec8082bd865dcb18c45bff323dc154edac',
            '6a0b45ddbade09caed2647e28fb1af8ea5c5068f3daa91d5ac100583a30cbe3b',
            'dd7f7d6c3b232e524a965a46cff59d9bbc58698f123af63e32aac6c53213d00e',
            '86c35335635950b110ca1b761c2cba856a5ba473e30fa6d33713cea6838a3959'
        }

        for file in self.__file_iterator__():
            logger.info(f'Processing {file}')
            with open(file, newline='') as csv_file:
                csv_reader = csv.DictReader(csv_file, delimiter=',')
                for row in csv_reader:
                    ms_name = row['msname']
                    ms_instance = row['msinstanceid']
                    timestamp = int(row['timestamp'])

                    if ms_name not in ms_map:
                        ms_map[ms_name] = {
                            'max_timestamp': timestamp,
                            'min_timestamp': timestamp,
                            'instances': {ms_instance}
                        }
                    else:
                        self.__check_microservice_duration__(row, ms_map)
                        self.__check_microservice_number_instances__(row, ms_map)
                    if ms_instance not in ms_instance_map:
                        ms_instance_map[ms_instance] = {
                            'max_timestamp': timestamp,
                            'min_timestamp': timestamp,
                            'cpu': [],
                            'memory': []
                        }
                    else:
                        self.__check_microservice_instance_duration__(row, ms_instance_map)

                    self.__check_microservice_time_series__(row, ms_follow, ms_instance_map)

                    # break
                    # count += 1
                    # if count > 500:
                    #    break
            # break
            # count += 1
            # if count > 1:
            #     break

        logging.info(f'{self.__class__}: Total number of microservices {len(ms_map)}')
        logging.info(f'{self.__class__}: Total number of microservices instances {len(ms_instance_map)}')

        plot_hist(
            False,
            f'{output_path}/microservices_durations',
            'Microservices durations',
            'durations in ms',
            'frequency',
            x=[value['max_timestamp'] - value['min_timestamp'] for value in ms_map.values()]
        )

        plot_hist(
            False,
            f'{output_path}/microservices_instances_durations',
            'Microservices instances durations',
            'durations in ms',
            'frequency',
            x=[value['max_timestamp'] - value['min_timestamp'] for value in ms_instance_map.values()]
        )

        plot_hist(
            False,
            f'{output_path}/microservices_number_instances',
            'Microservices number of instances',
            'instances',
            'frequency',
            x=[len(value['instances']) for value in ms_map.values()],
            bins=100
        )

        plot_hist(
            False,
            f'{output_path}/microservices_number_instances_lim_100',
            'Microservices number of instances',
            'instances',
            'frequency',
            x=[len(value['instances']) for value in ms_map.values() if len(value['instances']) < 100],
            bins=50
        )

        services_with_little_instances = [key for key, value in ms_map.items() if len(value['instances']) < 25]
        print(len(services_with_little_instances))

        for ms_name in ms_follow:
            lines = []
            for instance_id in ms_map[ms_name]['instances']:
                cpu = ms_instance_map[instance_id]['cpu']
                lines.append((
                    cpu,
                    list(range(len(cpu))),
                    f'cpu_{instance_id}'
                ))
            compare_multiple_lines(
                False,
                lines,
                'CPU',
                'time',
                f'CPU utilization of the instances of the service {ms_name}',
                f'{output_path}/cpu_utilization_instances_service_{ms_name}'
            )

    def get_number_features(self) -> int:
        raise NotImplementedError()

    def get_features_labels(self) -> List[str]:
        raise NotImplementedError()

    # ---> Auxiliary methods

    def __file_iterator__(self):
        ms_files = [
            '../data/alibaba_microservices_2021/MSResource/MSResource_0.csv',
            '../data/alibaba_microservices_2021/MSResource/MSResource_1.csv',
            '../data/alibaba_microservices_2021/MSResource/MSResource_2.csv',
            '../data/alibaba_microservices_2021/MSResource/MSResource_3.csv',
            '../data/alibaba_microservices_2021/MSResource/MSResource_4.csv',
            '../data/alibaba_microservices_2021/MSResource/MSResource_5.csv',
            '../data/alibaba_microservices_2021/MSResource/MSResource_6.csv',
            '../data/alibaba_microservices_2021/MSResource/MSResource_7.csv',
            '../data/alibaba_microservices_2021/MSResource/MSResource_8.csv',
            '../data/alibaba_microservices_2021/MSResource/MSResource_9.csv',
            '../data/alibaba_microservices_2021/MSResource/MSResource_10.csv',
            '../data/alibaba_microservices_2021/MSResource/MSResource_11.csv'
        ]
        # ms_files.sort()
        for file in ms_files:
            # tar = tarfile.open(file, "r:gz")
            # member = tar.getmembers()[0]
            # yield tar.extractfile(member)
            yield file

    def __check_microservice_duration__(self, row, ms_map: dict):
        ms_name = row['msname']
        timestamp = int(row['timestamp'])
        if ms_map[ms_name]['max_timestamp'] < timestamp:
            ms_map[ms_name]['max_timestamp'] = timestamp
        if ms_map[ms_name]['min_timestamp'] > timestamp:
            ms_map[ms_name]['min_timestamp'] = timestamp

    def __check_microservice_instance_duration__(self, row, ms_instance_map: dict):
        ms_instance = row['msinstanceid']
        timestamp = int(row['timestamp'])
        if ms_instance_map[ms_instance]['max_timestamp'] < timestamp:
            ms_instance_map[ms_instance]['max_timestamp'] = timestamp
        if ms_instance_map[ms_instance]['min_timestamp'] > timestamp:
            ms_instance_map[ms_instance]['min_timestamp'] = timestamp

    def __check_microservice_number_instances__(self, row, ms_map: dict):
        ms_name = row['msname']
        ms_instance = row['msinstanceid']
        if ms_instance not in ms_map[ms_name]['instances']:
            ms_map[ms_name]['instances'].add(ms_instance)

    def __check_microservice_time_series__(self, row, ms_follow: set, ms_instance_map: dict):
        ms_name = row['msname']
        ms_instance = row['msinstanceid']
        if ms_name in ms_follow:
            ms_instance_map[ms_instance]['cpu'] += [float(row['instance_cpu_usage'])]
            ms_instance_map[ms_instance]['memory'] += [float(row['instance_memory_usage'])]
