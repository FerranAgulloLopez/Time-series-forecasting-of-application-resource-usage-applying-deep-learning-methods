import csv
import json
import logging
import time

logger = logging.getLogger(__name__)


def timeit(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        logger.debug(f'{method.__name__}. Elapsed time {te - ts}')
        return result

    return timed


def load_json(path):
    with open(path) as file:
        data = json.load(file)
    return data


def save_json(path, data):
    path += '.json'
    with open(path, 'w') as file:
        json.dump(data, file, indent=2)


def print_pretty_json(data):
    logger.info(json.dumps(data, indent=4))


def load_csv(path):
    rows = []
    with open(path, newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',')
        for row in spamreader:
            rows.append(row)
    return rows


def save_csv(file, rows):
    with open(f'{file}.csv', 'w', newline='') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=',')
        for row in rows:
            spamwriter.writerow(row)
