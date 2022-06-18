import argparse
import logging
import random
import sys

import numpy as np
import torch
import torch.backends.cudnn as cudnn

from auxiliary_files.other_methods.util_functions import load_json, save_json, print_pretty_json
from factories.data_factory import DataFactory
from factories.model_factory import ModelFactory


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
    parser.add_argument('--redirect', default=False, type=parse_bool, help="Redirect output to output directory", required=False)
    args = parser.parse_args()
    
    # TODO check the first path is a file and a json
    # TODO check the second path is a directory
    # TODO check config file correctness
    
    return args


def main(config_file: str, output_path: str, visualize: bool):
    # visualize -> show output charts in standard output

    # load configuration file
    config = load_json(config_file)

    # setup logging
    stdout_handler = logging.StreamHandler(sys.stdout)
    logging.basicConfig(
        level=eval(f'logging.{config["logging"] if "logging" in config else "DEBUG"}'),
        format='[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s',
        handlers=[stdout_handler]
    )

    logger = logging.getLogger(__name__)
    
    # config setup
    if int(config['manual_seed']) == -1:
        config['manual_seed'] = random.randint(1, 10000)
    random.seed(config['manual_seed'])
    np.random.seed(config['manual_seed'])
    torch.manual_seed(config['manual_seed'])
    cudnn.benchmark = True
    torch.set_default_tensor_type(torch.FloatTensor)
    
    # load data
    logger.info('Loading data')
    data = DataFactory.select_data(config['data'], output_path, config['device'])
    data.show_info()
    data.load_data()

    # load model
    logger.info('Loading model')
    model = ModelFactory.select_model(config['model'], data, output_path, config['device'])
    model.show_info()

    # save initial config
    save_json(output_path + '/initial_config', config)

    # train and test model
    logger.info('Training and testing model')
    train_elapsed_time, test_elapsed_time = model.train_test()

    # save results and model
    if config['save_model']:
        logger.info('Saving model')
        model.save_model()
    model.save_results(visualize)
    
    # save elapsed times
    total_times_dict = {
        'train_elapsed_time': train_elapsed_time,
        'test_elapsed_time': test_elapsed_time
    }
    print_pretty_json(total_times_dict)
    save_json(output_path + '/total_times', total_times_dict)


if __name__ == '__main__':
    # Get input params (input config and output paths)
    args = parse_arguments()
    
    # Redirect program output
    if args.redirect:
        f = open(args.output + '/log.txt', 'w')
        sys.stdout = f
    
    # Run main program
    main(args.config, args.output, not args.redirect)
    if args.redirect:
        f.close()
