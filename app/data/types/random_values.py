import logging
import random
from typing import List

import torch
from torch import LongTensor
from torch.utils.data import DataLoader, TensorDataset

from app.data.data_type_abstract import DataTypeAbstract
from app.auxiliary_files.other_methods.util_functions import timeit

logger = logging.getLogger(__name__)


class DataTypeRandomValues(DataTypeAbstract):

    def __init__(self, config, *args):
        super().__init__(config, *args)
        self.shuffle = config['shuffle'] if 'shuffle' in config else True
        self.total_number = config['total_number']

    def create_data_loader(
            self,
            time_series_ids: List[str],
            time_series_lengths: List[int],
            is_train: bool
    ) -> DataLoader:
        # create data loader with the indexes of all possible values
        # we do not create a new array to not waste memory (the transformation is done in collate_fn)
        # the indexes are in the way -> (time_series_index, time_position)
        indexes = []
        for time_series_id_index, time_series_id in enumerate(time_series_ids):
            time_series_length = time_series_lengths[time_series_id_index]
            if time_series_length > self.total_number:
                random_sample = random.sample(range(
                    self.lag_size, time_series_length - self.prediction_size + 1),
                    self.total_number
                )
                possible_time_values = [(time_series_id, time_position) for time_position in random_sample]
            else:
                possible_time_values = [(time_series_id, time_position) for time_position in
                                        range(self.lag_size, time_series_length - self.prediction_size + 1)]
            indexes += possible_time_values
        logger.info(f'Total possible values {len(indexes)}')

        return DataLoader(
            dataset=TensorDataset(LongTensor(indexes)),
            shuffle=(is_train and self.shuffle),
            batch_size=self.batch_size,
            pin_memory=True,
            collate_fn=self.collate_fn
        )

    @timeit
    def collate_fn(self, samples):
        time_series_ids_dict = {}
        for sample in samples:
            time_series_id, time_position = sample[0][0].item(), sample[0][1].item()
            if time_series_id not in time_series_ids_dict:
                time_series_ids_dict[time_series_id] = [time_position]
            else:
                time_series_ids_dict[time_series_id] += [time_position]

        time_series_ids_list, _, _, time_series_values_values_list, time_series_target_values_list = self.data_source.load_time_series_list(
            list(time_series_ids_dict.keys()),
            times=False,
            init=False
        )

        values = torch.zeros((len(samples), time_series_values_values_list[0].shape[0], self.lag_size))  # TODO does not work if list empty
        target = torch.zeros(len(samples), self.prediction_size)

        count = 0
        ids = []
        for time_series_index, time_series_id in enumerate(time_series_ids_list):
            for time_position in time_series_ids_dict[time_series_id]:
                values[count] = torch.from_numpy(time_series_values_values_list[time_series_index][:, (time_position - self.lag_size):time_position])
                target[count] = torch.from_numpy(time_series_target_values_list[time_series_index][0, time_position:(time_position + self.prediction_size)])
                ids.append((time_series_id, time_position))
                count += 1

        return ids, values, target
