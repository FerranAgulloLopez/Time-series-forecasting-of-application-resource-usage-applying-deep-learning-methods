import random

from app.auxiliary_files.other_methods.util_functions import load_csv
from app.auxiliary_files.other_methods.visualize import compare_multiple_lines_matrix

INPUT_FILE_PATH = 'sample_time_series_<1d.csv'
LENGTH = '20d-25d'
NUMBER = 3


def main():
    rows = load_csv(INPUT_FILE_PATH)

    data = {}
    header = rows[0]
    for row_index in range(1, len(rows)):
        row = rows[row_index]
        _id = f'{row[0]}_{row[1]}'
        if _id in data:
            for data_index in range(2, len(header)):
                data[_id][header[data_index]] += [float(row[data_index])]
        else:
            data[_id] = {header[data_index]: [float(row[data_index])] for data_index in range(2, len(header))}

    while len(data) > NUMBER:
        random_key = random.choice(list(data.keys()))
        del data[random_key]


    features_labels = ['cpu_average_usage', 'cpu_max_usage']
    charts = []
    for time_series_id, time_series_values in data.items():
        charts.append((
            [(
                feature_values,
                time_series_values['time_stamp'],
                feature_key
            ) for feature_key, feature_values in time_series_values.items() if feature_key in features_labels],
            'y',
            f''
        ))

    compare_multiple_lines_matrix(
        False,
        charts,
        '',
        'time',
        f'{INPUT_FILE_PATH.replace(".csv", "")}_cpu',
        ncols=1
    )

    features_labels = ['memory_average_usage', 'memory_max_usage']
    charts = []
    for time_series_id, time_series_values in data.items():
        charts.append((
            [(
                feature_values,
                time_series_values['time_stamp'],
                feature_key
            ) for feature_key, feature_values in time_series_values.items() if feature_key in features_labels],
            'y',
            f''
        ))

    compare_multiple_lines_matrix(
        False,
        charts,
        '',
        'time',
        f'{INPUT_FILE_PATH.replace(".csv", "")}_memory',
        ncols=1
    )


if __name__ == '__main__':
    main()
