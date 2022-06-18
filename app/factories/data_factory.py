from app.data.data_source_abstract import DataSourceAbstract
from app.data.data_type_interface import DataTypeInterface
from app.data.sources.alibaba_microservices_2021 import DataSourceAlibabaMicroservices2021
from app.data.sources.google_2019 import DataSourceGoogle2019
from app.data.sources.alibaba_2018 import DataSourceAlibaba2018
from app.data.types.random_values import DataTypeRandomValues
from app.data.types.select_values_unbiased import DataTypeSelectValuesUnbiased
from app.data.types.full_values_unbiased_intervals import DataTypeFullValuesUnbiasedIntervals
from app.data.types.full_values_unbiased_target_max import DataTypeFullValuesUnbiasedTargetMax
from app.data.types.full_values_unbiased_intervals_transformer import DataTypeFullValuesUnbiasedIntervalsTransformer
from app.data.types.balanced_values_unbiased_intervals_transformer import DataTypeBalancedValuesUnbiasedIntervalsTransformer
from app.data.types.full_values_unbiased import DataTypeFullValuesUnbiased
from app.data.types.balanced_values_unbiased_intervals import DataTypeBalancedValuesUnbiasedIntervals
from app.data.types.select_values_unbiased_target_max import DataTypeSelectValuesUnbiasedTargetMax
from app.data.types.balanced_values_unbiased import DataTypeBalancedValuesUnbiased
from app.data.types.simple_time_series import DataTypeSimpleTimeSeries
from app.data.types.data_process import DataTypeProcess


class DataFactory:

    def __init__(self):
        raise Exception('This class can not be instantiated')

    @staticmethod
    def select_data(config: dict, output_path: str, device: str) -> DataTypeInterface:
        source = DataFactory.select_data_source(config['source'], device)
        return DataFactory.select_data_type(config['type'], source, output_path, device)

    @staticmethod
    def select_data_source(config, *args) -> DataSourceAbstract:
        name = config['name']
        if name == 'alibaba_microservices_2021':
            data = DataSourceAlibabaMicroservices2021(config, *args)
        elif name == 'google_2019':
            data = DataSourceGoogle2019(config, *args)
        elif name == 'alibaba_2018':
            data = DataSourceAlibaba2018(config, *args)
        else:
            raise Exception('The data source with name ' + name + ' does not exist')
        if issubclass(type(data), DataSourceAbstract):
            return data
        else:
            raise Exception('The data source does not follow the interface definition')

    @staticmethod
    def select_data_type(config, *args) -> DataTypeInterface:
        name = config['name']
        if name == 'full_values_unbiased':
            data = DataTypeFullValuesUnbiased(config, *args)
        elif name == 'balanced_values_unbiased':
            data = DataTypeBalancedValuesUnbiased(config, *args)
        elif name == 'full_values_unbiased_target_max':
            data = DataTypeFullValuesUnbiasedTargetMax(config, *args)
        elif name == 'full_values_unbiased_intervals':
            data = DataTypeFullValuesUnbiasedIntervals(config, *args)
        elif name == 'balanced_values_unbiased_intervals':
            data = DataTypeBalancedValuesUnbiasedIntervals(config, *args)
        elif name == 'full_values_unbiased_intervals_transformer':
            data = DataTypeFullValuesUnbiasedIntervalsTransformer(config, *args)
        elif name == 'balanced_values_unbiased_intervals_transformer':
            data = DataTypeBalancedValuesUnbiasedIntervalsTransformer(config, *args)
        elif name == 'random_values':
            data = DataTypeRandomValues(config, *args)
        elif name == 'select_values_unbiased':
            data = DataTypeSelectValuesUnbiased(config, *args)
        elif name == 'select_values_unbiased_target_max':
            data = DataTypeSelectValuesUnbiasedTargetMax(config, *args)
        elif name == 'data_preprocess':
            data = DataTypeProcess(config, *args)
        elif name == 'simple_time_series':
            data = DataTypeSimpleTimeSeries(config, *args)
        else:
            raise Exception('The data type with name ' + name + ' does not exist')
        if issubclass(type(data), DataTypeInterface):
            return data
        else:
            raise Exception('The data type does not follow the interface definition')
