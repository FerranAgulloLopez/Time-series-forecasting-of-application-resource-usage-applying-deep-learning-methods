import numpy as np

from app.auxiliary_files.other_methods.util_functions import load_json, save_json
from app.auxiliary_files.other_methods.visualize import plot_hist


def main():
    containers_request_usage = load_json('./containers_cpu_request.json')
    output_path = '.'

    values, counts = np.unique(list(containers_request_usage.values()), return_counts=True)
    plot_hist(
        False,
        f'{output_path}/requested_cpu_usage_hist',
        'Histogram of the requested cpu usage',
        'requested cpu usage',
        'frequency',
        x=containers_request_usage.values(),
        bins=10
    )
    save_json(f'{output_path}/requested_cpu_usage_hist', {'values': [int(x) for x in list(values)], 'counts': [int(x) for x in list(counts)]})


if __name__ == '__main__':
    main()
