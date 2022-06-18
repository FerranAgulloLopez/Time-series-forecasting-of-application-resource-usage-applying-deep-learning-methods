import json
import logging
from time import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from app.auxiliary_files.model_methods.loss_functions import select_loss_function
from app.auxiliary_files.model_methods.model_operations import model_arq_to_json
from app.auxiliary_files.model_methods.model_preprocessing import preprocess_model
from app.auxiliary_files.model_methods.optimizers import select_optimizer
from app.auxiliary_files.other_methods.util_functions import print_pretty_json, save_json
from app.auxiliary_files.other_methods.visualize import compare_multiple_lines, compare_multiple_lines_points_color_extended
from app.models.model_type_interface import ModelTypeInterface

logger = logging.getLogger(__name__)


# define the default workflow for neural networks training
class ModelTypeNeuralNetwork(ModelTypeInterface):

    def __init__(self, config: dict, data_model, network: nn.Module, output_path: str, device: str):
        super().__init__(config, data_model, network, output_path, device)

        # save input params
        self.config = config
        self.data_model = data_model
        self.network = network
        self.output_path = output_path
        self.device = device

        # read config
        self.number_epochs = config['train_info']['number_epochs']

        # load components
        self.optimizer = select_optimizer(config['train_info']['optimizer'], self.network)
        self.loss_function = select_loss_function(config['train_info']['loss_function'], self.device)

        # prepare network
        if 'pretrained' in config:
            self.network.load_state_dict(torch.load(config['pretrained'], map_location=device))
        else:
            preprocess_model(config['transforms'], self.network)  # TODO check it does its job

        self.network = self.network.float()
        self.network = self.network.to(self.device)

        # auxiliary variables
        self.train_loss = torch.zeros((self.number_epochs, 1)).to(self.device).detach()
        self.val_loss = torch.zeros((self.number_epochs, 1)).to(self.device).detach()

    # ---> Main methods

    def show_info(self):
        logger.info(f'Model architecture: {json.dumps(model_arq_to_json(self.network), indent=2)}')

    def train_test(self):
        # -> train model
        init_time = time()

        # obtain data loaders
        train_data_loader = self.data_model.get_train_data_loader(for_training=True)
        val_data_loader = self.data_model.get_val_data_loader()

        # train network
        self.__train_network(train_data_loader, val_data_loader)
        total_train_time = time() - init_time

        # show training evolution
        losses = [
            (self.train_loss.to('cpu').numpy(), np.arange(self.train_loss.shape[0]), 'train loss'),
            (self.val_loss.to('cpu').numpy(), np.arange(self.val_loss.shape[0]), 'val loss')
        ]
        compare_multiple_lines(False, losses, 'Loss', 'epoch', 'Train evolution', self.output_path + '/train_evolution')
        best_train_scores_dict = {
            'best_model_losses': {
                'best_train_loss': {
                    'value': float(torch.min(self.train_loss).to('cpu').item()),
                    'epoch': int(torch.argmin(self.train_loss).to('cpu').item())
                },
                'best_val_loss': {
                    'value': float(torch.min(self.val_loss).to('cpu').item()),
                    'epoch': int(torch.argmin(self.val_loss).to('cpu').item()),
                    'corresponding_train_loss': float(self.train_loss[int(torch.argmin(self.val_loss).to('cpu').item())].to('cpu').item())  # TODO refactor
                }
            }
        }
        print_pretty_json(best_train_scores_dict)
        save_json(self.output_path + '/best_train_scores.json', best_train_scores_dict)

        # -> predict train, val and test splits
        init_time = time()

        # obtain test data loader
        test_data_loader = self.data_model.get_test_data_loader()

        # recreate train data loader without shuffling
        train_data_loader = self.data_model.get_train_data_loader(for_training=False)

        # compute loss
        train_loss = self.__compute_loss(train_data_loader)
        val_loss = self.__compute_loss(val_data_loader)
        test_loss = self.__compute_loss(test_data_loader)

        # predict
        train_data_predictions = self.__predict_network(train_data_loader)
        val_data_predictions = self.__predict_network(val_data_loader)
        test_data_predictions = self.__predict_network(test_data_loader)

        total_test_time = time() - init_time
        final_model_losses_dict = {
            'final_model_losses': {
                'final_train_loss': train_loss,
                'final_val_loss': val_loss,
                'final_test_loss': test_loss
            }
        }
        print_pretty_json(final_model_losses_dict)
        save_json(self.output_path + '/final_model_losses.json', final_model_losses_dict)

        # -> do visualization
        self.data_model.perform_train_output_visualization(train_data_predictions)
        self.data_model.perform_val_output_visualization(val_data_predictions)
        self.data_model.perform_test_output_visualization(test_data_predictions)

        # -> do evaluation
        train_final_predictions_ids, train_initial_traces, train_final_predictions = self.data_model.get_final_predictions(train_data_predictions)
        val_final_predictions_ids, val_initial_traces, val_final_predictions = self.data_model.get_final_predictions(val_data_predictions)
        test_final_predictions_ids, test_initial_traces, test_final_predictions = self.data_model.get_final_predictions(test_data_predictions)

        train_evaluation = self.__compute_evaluation(train_final_predictions_ids, train_initial_traces, train_final_predictions)
        val_evaluation = self.__compute_evaluation(val_final_predictions_ids, val_initial_traces, val_final_predictions)
        test_evaluation = self.__compute_evaluation(test_final_predictions_ids, test_initial_traces, test_final_predictions)
        print_pretty_json(train_evaluation)
        print_pretty_json(val_evaluation)
        print_pretty_json(test_evaluation)
        save_json(self.output_path + '/train_evaluation.json', train_evaluation)
        save_json(self.output_path + '/val_evaluation.json', val_evaluation)
        save_json(self.output_path + '/test_evaluation.json', test_evaluation)

        return total_train_time, total_test_time

    def save_results(self, visualize):
        pass

    def save_model(self):
        self.network = self.network.to('cpu')
        torch.save(self.network.state_dict(), self.output_path + '/network_weights.pt')
        self.network = self.network.to(self.device)

    # --> Auxiliary methods

    def __train_network(self, train_data_loader: DataLoader, val_data_loader: DataLoader):
        count_not_increased_val = 0

        for number_epoch in range(self.number_epochs):
            t = time()
            self.__train_epoch(number_epoch, train_data_loader, self.train_loss)
            train_time = time() - t
            t = time()

            self.__not_train_epoch(number_epoch, val_data_loader, self.val_loss)
            val_time = time() - t

            logger.info(
                str('====> Epoch: {} Train set loss: {:.6f}; time {}. Val set loss: {:.6f}; time: {} \n').format(
                    number_epoch,
                    self.train_loss[number_epoch][0],
                    train_time,
                    self.val_loss[number_epoch][0],
                    val_time
                    )
                )

            if number_epoch > 5 and self.val_loss[number_epoch] > torch.mean(self.val_loss[(number_epoch - 5):number_epoch]):
                count_not_increased_val += 1
                if count_not_increased_val >= 5:
                    self.train_loss = self.train_loss[:(number_epoch + 1)]
                    self.val_loss = self.val_loss[:(number_epoch + 1)]
                    return
            else:
                count_not_increased_val = 0

    def __compute_loss(self, data_loader: DataLoader):
        loss = torch.zeros((1, 1)).detach().to(self.device)
        self.__not_train_epoch(0, data_loader, loss)
        return loss.to('cpu').item()

    def __compute_evaluation(self, traces_ids, traces_values, traces_predictions):
        output = {}
        WINDOW_SIZE = 5

        lag_size = self.data_model.get_lag_size()
        prediction_size = self.data_model.get_prediction_size()

        over_provisioning = []
        under_provisioning = []
        spikes_precision = []
        spikes_recall = []
        spikes_recall_different = []

        for time_series_id_index, time_series_id in enumerate(traces_ids):
            values = traces_values[time_series_id_index]
            predictions = traces_predictions[time_series_id_index]

            min_values = np.min(values)

            values = values + np.abs(min_values)
            predictions = predictions + np.abs(min_values)

            _range = range(lag_size, values.shape[0] - prediction_size + 1)
            sum_values = np.sum(values)

            # over-provisioning
            difference = sum([predictions[time_position] - values[time_position]
                              if predictions[time_position] > values[time_position]
                              else 0
                              for time_position in _range])
            over_provisioning.append(difference / sum_values)

            # under-provisioning
            difference = sum([values[time_position] - predictions[time_position]
                              if values[time_position] > predictions[time_position]
                              else 0
                              for time_position in _range])
            under_provisioning.append(difference / sum_values)

            # spikes
            values_std = np.std(values)

            # spikes to detect
            values_spikes = []
            last_spike_value = None
            for time_position in _range:
                lag_mean = np.mean(values[(time_position - WINDOW_SIZE):time_position])
                if (values[time_position] - lag_mean) > (1 * values_std):
                    if last_spike_value is None:
                        values_spikes.append(time_position)
                        last_spike_value = values[time_position]
                    else:
                        difference = np.abs(values[time_position] - last_spike_value)
                        if difference > values[time_position] * 0.1:
                            values_spikes.append(time_position)
                            last_spike_value = values[time_position]
                else:
                    last_spike_value = None

            # """
            if time_series_id_index == 0:
                compare_multiple_lines_points_color_extended(
                    False,
                    [
                        (
                            values[500:1000],
                            np.arange(values.shape[0])[500:1000],
                            '#4f7cac',
                            3,
                            'real usage'
                        ),
                        (
                            predictions[500:1000],
                            np.arange(predictions.shape[0])[500:1000],
                            '#371e30',
                            3,
                            'resource allocation'
                        )
                    ],
                    [(values[time_position], time_position, "red") for time_position in values_spikes if
                     500 < time_position < 1000],
                    'y',
                    'time',
                    '',
                    f'{self.output_path}/values_spikes_{time_series_id}'
                )
            # """

            # detected spikes
            prediction_spikes = []
            last_spike_value = None
            for time_position in _range:
                lag_mean = np.mean(predictions[(time_position - WINDOW_SIZE):time_position])
                if (predictions[time_position] - lag_mean) > (1 * values_std):
                    if last_spike_value is None:
                        prediction_spikes.append(time_position)
                        last_spike_value = predictions[time_position]
                    else:
                        difference = np.abs(predictions[time_position] - last_spike_value)
                        if difference > values[time_position] * 0.1:
                            prediction_spikes.append(time_position)
                            last_spike_value = predictions[time_position]
                else:
                    last_spike_value = None

            # """
            if time_series_id_index == 0:
                compare_multiple_lines_points_color_extended(
                    False,
                    [
                        (
                            values[500:1000],
                            np.arange(values.shape[0])[500:1000],
                            '#4f7cac',
                            3,
                            'real usage'
                        ),
                        (
                            predictions[500:1000],
                            np.arange(predictions.shape[0])[500:1000],
                            '#371e30',
                            3,
                            'resource allocation'
                        )
                    ],
                    [(predictions[time_position], time_position, "red") for time_position in prediction_spikes if
                     500 < time_position < 1000],
                    'y',
                    'time',
                    '',
                    f'{self.output_path}/predictions_spikes_{time_series_id}'
                )
            # """

            # time intersection
            values_spikes_set = set(values_spikes)
            time_intersection_near = []
            for prediction_spike in prediction_spikes:
                found = False
                time_position = prediction_spike
                while not found and time_position < (prediction_spike + 5):
                    if time_position in values_spikes_set:
                        found = True
                        time_intersection_near.append((prediction_spike, time_position))
                    time_position += 1

                if not found:
                    time_position = prediction_spike - 5
                    while not found and time_position < prediction_spike:
                        if time_position in values_spikes_set:
                            found = True
                            time_intersection_near.append((prediction_spike, time_position))
                        time_position += 1

            if time_series_id_index == 0:
                compare_multiple_lines_points_color_extended(
                    False,
                    [
                        (
                            values[500:1000],
                            np.arange(values.shape[0])[500:1000],
                            '#4f7cac',
                            3,
                            'real usage'
                        ),
                        (
                            predictions[500:1000],
                            np.arange(predictions.shape[0])[500:1000],
                            '#371e30',
                            3,
                            'resource allocation'
                        )
                    ],
                    [(predictions[prediction_spike], prediction_spike, "red") for (prediction_spike, value_spike) in
                     time_intersection_near if 500 < prediction_spike < 1000],
                    'y',
                    'time',
                    '',
                    f'{self.output_path}/time_intersection_spikes_near_{time_series_id}'
                )

            # height intersection near
            space_intersection_near = []
            for (prediction_spike, value_spike) in time_intersection_near:
                difference = predictions[prediction_spike] - values[value_spike]
                if difference <= 0 and np.abs(difference) < (values[value_spike] * 0.05):
                    space_intersection_near.append((prediction_spike, value_spike))
                elif difference > 0 and np.abs(difference) < (values[value_spike] * 0.15):
                    space_intersection_near.append((prediction_spike, value_spike))

            if time_series_id_index == 0:
                compare_multiple_lines_points_color_extended(
                    False,
                    [
                        (
                            values[500:1000],
                            np.arange(values.shape[0])[500:1000],
                            '#4f7cac',
                            3,
                            'real usage'
                        ),
                        (
                            predictions[500:1000],
                            np.arange(predictions.shape[0])[500:1000],
                            '#371e30',
                            3,
                            'resource allocation'
                        )
                    ],
                    [(predictions[prediction_spike], prediction_spike, "red") for (prediction_spike, value_spike) in
                     space_intersection_near if 500 < prediction_spike < 1000],
                    'y',
                    'time',
                    '',
                    f'{self.output_path}/space_intersection_spikes_near_{time_series_id}'
                )

            # print(f'values_spikes: {len(values_spikes)}; predictions_spikes: {len(prediction_spikes)}')
            # print(values_spikes)
            # print(prediction_spikes)
            # print(time_intersection_near)
            # print(space_intersection_near)

            correctly_predicted_spikes = []
            for value_spike in values_spikes:
                found = False
                time_position = value_spike - 2
                while not found and time_position <= value_spike:
                    difference = predictions[time_position] - values[value_spike]
                    if difference <= 0 and np.abs(difference) < (values[value_spike] * 0.05):
                        found = True
                        correctly_predicted_spikes.append(value_spike)
                    elif difference > 0 and np.abs(difference) < (values[value_spike] * 0.15):
                        found = True
                        correctly_predicted_spikes.append(value_spike)
                    time_position += 1

            correctly_predicted_spikes_different = []
            for value_spike in values_spikes:
                found = False
                time_position = value_spike - 2
                while not found and time_position <= value_spike:
                    difference = predictions[time_position] - values[value_spike]
                    if difference >= 0:
                        found = True
                        correctly_predicted_spikes_different.append(value_spike)
                    time_position += 1

            if time_series_id_index == 0:
                compare_multiple_lines_points_color_extended(
                    False,
                    [
                        (
                            values[500:1000],
                            np.arange(values.shape[0])[500:1000],
                            '#4f7cac',
                            3,
                            'real usage'
                        ),
                        (
                            predictions[500:1000],
                            np.arange(predictions.shape[0])[500:1000],
                            '#371e30',
                            3,
                            'resource allocation'
                        )
                    ],
                    [(values[value_spike], value_spike, "red") for value_spike in
                     correctly_predicted_spikes if 500 < value_spike < 1000],
                    'y',
                    'time',
                    '',
                    f'{self.output_path}/corrected_predicted_spikes_{time_series_id}'
                )

            close_predicted_spikes = {value_spike for (prediction_spike, value_spike) in space_intersection_near}

            if len(prediction_spikes) > 0:
                spikes_precision.append(len(close_predicted_spikes) / len(prediction_spikes))
            if len(values_spikes) > 0:
                spikes_recall.append(len(correctly_predicted_spikes) / len(values_spikes))
                spikes_recall_different.append(len(correctly_predicted_spikes_different) / len(values_spikes))

        output['over-provisioning'] = sum(over_provisioning) / len(over_provisioning)
        output['under-provisioning'] = sum(under_provisioning) / len(under_provisioning)
        if len(spikes_precision) > 0:
            output['spikes_precision'] = sum(spikes_precision) / len(spikes_precision)
        else:
            output['spikes_precision'] = 0
        if len(spikes_recall):
            output['spikes_recall'] = sum(spikes_recall) / len(spikes_recall)
        else:
            output['spikes_recall'] = 0
        if len(spikes_recall_different) > 0:
            output['spikes_recall_different'] = sum(spikes_recall_different) / len(spikes_recall_different)
        else:
            output['spikes_recall_different'] = 0

        return output


    def __predict_network(self, data_loader: DataLoader):
        self.network.eval()

        all_predictions = []

        with torch.no_grad():
            for index, (ids, values, _) in enumerate(data_loader, 0):  # iterate data loader
                values = values.to(self.device)
                prediction = self.network.predict(values)

                prediction = prediction.detach().to('cpu').numpy()
                all_predictions.append((ids, prediction))

        return all_predictions

    def __train_epoch(self, number_epoch: int, train_data_loader: DataLoader, losses_array: torch.Tensor):
        self.network.train()

        for index, (_, train_values, train_target) in enumerate(train_data_loader, 0):  # iterate data loader
            ts = time()

            train_values = train_values.to(self.device)
            train_target = train_target.to(self.device)

            self.optimizer.zero_grad()
            train_output = self.network(train_values, train_target)

            te = time()
            logger.debug(f'__train_epoch_1. Elapsed time {te - ts}')

            ts = time()

            loss = self.loss_function.run(train_output, train_target)
            loss.backward()
            self.optimizer.step()

            losses_array[number_epoch][0] = losses_array[number_epoch][0].add(loss.detach().view(1))  # update loss array

            te = time()
            logger.debug(f'__train_epoch_2. Elapsed time {te - ts}')

        losses_array[number_epoch][0] = losses_array[number_epoch][0].div(len(train_data_loader))  # update loss array

    def __not_train_epoch(self, number_epoch: int, data_loader: DataLoader, losses_array: torch.Tensor):
        self.network.eval()

        with torch.no_grad():
            for index, (_, values, target) in enumerate(data_loader, 0):  # iterate data loader
                values = values.to(self.device)
                target = target.to(self.device)

                output = self.network(values, target)
                loss = self.loss_function.run(output, target)

                losses_array[number_epoch][0] = losses_array[number_epoch][0].add(loss.detach().view(1))  # update loss array

        losses_array[number_epoch][0] = losses_array[number_epoch][0].div(len(data_loader))  # update loss array
