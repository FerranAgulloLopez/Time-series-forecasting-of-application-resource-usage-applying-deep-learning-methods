from app.auxiliary_files.model_methods.networks import select_network
from app.models.model_type_interface import ModelTypeInterface
from app.models.types.model_type_neural_network import ModelTypeNeuralNetwork
from app.models.types.model_type_neural_network_transformer import ModelTypeNeuralNetworkTransformer
from app.models.types.model_type_classic_method import ModelTypeClassicMethod


class ModelFactory:

    def __init__(self):
        raise Exception('This class can not be instantiated')

    @staticmethod
    def select_model(config, data_model, output_path, device) -> ModelTypeInterface:
        network = ModelFactory.select_model_network(
            config['network'],
            data_model.get_lag_size(),
            data_model.get_prediction_size(),
            data_model.get_number_features(),
            device
        )
        return ModelFactory.select_model_type(config['type'], data_model, network, output_path, device)

    @staticmethod
    def select_model_network(config, lag_size, prediction_size, number_features, device):  # TODO create interface for networks
        return select_network(config, lag_size, prediction_size, number_features, device)

    @staticmethod
    def select_model_type(config, *args) -> ModelTypeInterface:
        name = config['name']
        if name == 'neural_network':
            model = ModelTypeNeuralNetwork(config, *args)
        elif name == 'transformer':
            model = ModelTypeNeuralNetworkTransformer(config, *args)
        elif name == 'classic_method':
            model = ModelTypeClassicMethod(config, *args)
        else:
            raise Exception('The model type with name ' + name + ' does not exist')
        if issubclass(type(model), ModelTypeInterface):
            return model
        else:
            raise Exception('The model type does not follow the interface definition')
