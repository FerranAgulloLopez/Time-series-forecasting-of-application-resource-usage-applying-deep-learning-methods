from os import listdir
import numpy as np
from os.path import isfile, exists, join


INPUT_FOLDER_PATH = '/home/ferran/Documents/bsc/time_series/workloads_resource_prediction/data/alibaba_2018/data/extracted_data'
FEATURE_INDEX = 8  # disk_io_percent

input_files = [
    filename for filename in listdir(INPUT_FOLDER_PATH)
    if isfile(join(INPUT_FOLDER_PATH, filename))
    and filename.endswith('.npy')
]

count = 0
for input_file in input_files:
    print(f'Left: {len(input_files) - count}')
    count += 1
    time_series_values = np.load(join(INPUT_FOLDER_PATH, input_file))
    meh = time_series_values[FEATURE_INDEX]
    meh2 = time_series_values[FEATURE_INDEX] > 100
    time_series_values[FEATURE_INDEX] = np.where((time_series_values[FEATURE_INDEX] < 0) | (time_series_values[FEATURE_INDEX] > 100), np.nan, time_series_values[FEATURE_INDEX])
    np.save(join(INPUT_FOLDER_PATH, input_file), time_series_values)
