import matplotlib.pyplot as plt
import numpy as np

from app.auxiliary_files.other_methods.util_functions import load_csv


def main():
    plt.rcParams.update({'font.size': 20})

    rows = load_csv('chart_requested_used_comparison.csv')

    length_order = [
        '<1d',
        '1d-2d',
        '2d-3d',
        '3d-5d',
        '5d-10d',
        '10d-15d',
        '15d-20d',
        '20d-25d',
        '>25d'
    ]

    data = {}
    header = rows[0]
    for row_index in range(1, len(rows)):
        row = rows[row_index]
        data[row[0]] = {header[data_index]: float(row[data_index]) for data_index in range(1, len(header))}

    chart_features = ['avg_requested_cpu', 'avg_cpu_average_usage', 'avg_cpu_max_usage']
    plt.subplots(figsize=(20, 10))
    bar_width = 0.25
    pos = np.arange(len(length_order))
    for feature in chart_features:
        plt.bar(
            x=pos,
            height=[data[length][feature] for length in length_order],
            width=bar_width
        )
        pos = [x + bar_width for x in pos]
    plt.xlabel('length')
    plt.ylabel('CPU usage')
    plt.title('')
    plt.legend(labels=['average requested cpu', 'average cpu average usage', 'average cpu maximum usage'])
    plt.xticks([r + bar_width for r in range(len(length_order))], length_order)
    plt.grid(True)
    plt.savefig('chart_cpu_requested_used_comparison', bbox_inches='tight')

    print('percentages cpu average/max', [(length, data[length]['avg_cpu_average_usage'] / data[length]['avg_cpu_max_usage']) for length in length_order])


    chart_features = ['avg_requested_memory', 'avg_memory_average_usage', 'avg_memory_max_usage']
    plt.subplots(figsize=(20, 10))
    bar_width = 0.25
    pos = np.arange(len(length_order))
    for feature in chart_features:
        plt.bar(
            x=pos,
            height=[data[length][feature] for length in length_order],
            width=bar_width
        )
        pos = [x + bar_width for x in pos]
    plt.xlabel('length')
    plt.ylabel('memory usage')
    plt.title('')
    plt.legend(labels=['average requested memory', 'average memory average usage', 'average memory maximum usage'])
    plt.xticks([r + bar_width for r in range(len(length_order))], length_order)
    plt.grid(True)
    plt.savefig('chart_memory_requested_used_comparison', bbox_inches='tight')

    print('percentages memory average/max', [(length, data[length]['avg_memory_average_usage'] / data[length]['avg_memory_max_usage']) for length in length_order])


if __name__ == '__main__':
    main()
