import argparse
import logging
import random
import sys

import numpy as np
import torch
import torch.backends.cudnn as cudnn

from auxiliary_files.other_methods.util_functions import load_json, save_json
from factories.data_factory import DataFactory

stdout_handler = logging.StreamHandler(sys.stdout)
logging.basicConfig(
    level=logging.DEBUG,
    format='[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s',
    handlers=[stdout_handler]
)

logger = logging.getLogger(__name__)


def parse_arguments():
    def parse_bool(s: str):
        if s.casefold() in ['1', 'true', 'yes']:
            return True
        if s.casefold() in ['0', 'false', 'no']:
            return False
        raise ValueError()

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--config', type=str, help="Path to config file", required=True)
    parser.add_argument('--output', type=str, help="Path to output directory", required=True)
    parser.add_argument('--redirect', default=False, type=parse_bool, help="Redirect output to output directory",
                        required=False)
    args = parser.parse_args()

    # TODO check the first path is a file and a json
    # TODO check the second path is a directory
    # TODO check config file correctness

    return args


def main(config_file: str, output_path: str):
    # visualize -> show output charts in standard output
    logging.basicConfig(level=logging.DEBUG)

    # Load configuration file
    config = load_json(config_file)
    save_json(output_path + '/initial_config', config)

    # Config setup
    if int(config['manual_seed']) == -1:
        config['manual_seed'] = random.randint(1, 10000)
    random.seed(config['manual_seed'])
    np.random.seed(config['manual_seed'])
    torch.manual_seed(config['manual_seed'])
    cudnn.benchmark = True

    # Load data
    logger.info('Loading data')
    data = DataFactory.select_data(config['data'], output_path, config['device'])
    data.show_info()
    data.load_data()

    logger.info('\nGenerating visualizations')
    # data.visualize(output_path)

    # Save config
    save_json(output_path + '/config', config)


if __name__ == '__main__':

    # Get input params (input config and output paths)
    args = parse_arguments()

    # Redirect program output
    if args.redirect:
        f = open(args.output + '/log.txt', 'w')
        sys.stdout = f

    # Run main program
    main(args.config, args.output)
    if args.redirect:
        f.close()
