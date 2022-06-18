import matplotlib.pyplot as plt
import numpy as np

from app.auxiliary_files.other_methods.util_functions import load_csv


def main():
    plt.rcParams.update({'font.size': 20})

    rows = load_csv('chart_priorities.csv')

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

    data = {length: {'monitoring_tier': 0, 'production_tier': 0, 'mid_iter': 0, 'best_effort': 0, 'free_tier': 0} for length in length_order}
    header = rows[0]
    for row_index in range(1, len(rows)):
        row = rows[row_index]
        priority_number = int(row[1])
        priority_frequency = int(row[2])
        priority_name = ''
        if priority_number >= 360:
            priority_name = 'monitoring_tier'
        elif priority_number >= 120:
            priority_name = 'production_tier'
        elif priority_number >= 116:
            priority_name = 'mid_iter'
        elif priority_number >= 100:
            priority_name = 'best_effort'
        elif priority_number >= 0:
            priority_name = 'free_tier'
        else:
            raise Exception('Not recognized priority level')

        data[row[0]][priority_name] += priority_frequency

    for length in length_order:
        total_number = sum([value for value in data[length].values()])
        data[length] = {key: value / total_number for key, value in data[length].items()}

    chart_features = ['monitoring_tier', 'production_tier', 'mid_iter', 'best_effort', 'free_tier']
    plt.subplots(figsize=(25, 10))
    bar_width = 0.15
    pos = np.arange(len(length_order))
    for feature in chart_features:
        plt.bar(
            x=pos,
            height=[data[length][feature] for length in length_order],
            width=bar_width,

        )
        pos = [x + bar_width for x in pos]
    plt.xlabel('length')
    plt.ylabel('percentage')
    plt.title('')
    plt.legend(labels=['monitoring tier', 'production tier', 'mid tier', 'best effort tier', 'free tier'])
    plt.xticks([r + bar_width for r in range(len(length_order))], length_order)
    plt.grid(True)
    plt.savefig('priorities_per_length', bbox_inches='tight')


if __name__ == '__main__':
    main()
