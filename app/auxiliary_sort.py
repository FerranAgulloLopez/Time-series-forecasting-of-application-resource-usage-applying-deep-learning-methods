from os import listdir
import numpy as np
from os.path import isfile, exists, join


INPUT_FOLDER_PATH = '/home/ferran/Documents/bsc/time_series/workloads_resource_prediction/data/alibaba_2018/data/extracted_data'
TIME_FEATURE_INDEX = 0

input_files = [
    filename for filename in listdir(INPUT_FOLDER_PATH)
    if isfile(join(INPUT_FOLDER_PATH, filename))
    and filename.endswith('.npy')
]

is_sorted = lambda a: np.all(a[:-1] <= a[1:])
count = 0
not_in_order_count = 0
for input_file in input_files:
    print(f'Left: {len(input_files) - count}')
    count += 1
    time_series_values = np.load(join(INPUT_FOLDER_PATH, input_file))
    if not is_sorted(time_series_values[TIME_FEATURE_INDEX, :]):
        not_in_order_count += 1
        time_series_values = time_series_values[:, time_series_values[TIME_FEATURE_INDEX, :].argsort()]
        np.save(join(INPUT_FOLDER_PATH, input_file), time_series_values)


print(f'Number of time series not in order: {not_in_order_count}')



