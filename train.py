"""The train implementation.
"""
import os
import json
import argparse

from utils.log import get_logger
from utils.check import check_cfg
from utils.environment import set_deterministic_option, set_world_size
from tools.train_module import train_module


def parse_args():
    """The operation for parse arguments.

    Returns:
        argparse.Namespace : The arguments.
    
    Notice:
        We only use json format config file.
    """

    parser = argparse.ArgumentParser(description='Pytorch Imagenet Train')
    parser.add_argument(
        '--model_config_dir', default='/home/sckim/AI_Framework/configs/classification/alexnet/alexnet.json', help='model config path')
    parser.add_argument(
        '--data_config_dir', default='/home/sckim/AI_Framework/configs/classification/data_config.json', help='data config path')
    parser.add_argument(
        '--env_config_dir', default='/home/sckim/AI_Framework/configs/env_config.json', help='environment config path')
    args = parser.parse_args()

    return args


def main():
    """The operation for main.
    """
    #load config
    args = parse_args()

    with open(args.model_config, 'r') as f:
        model_cfg = json.load(f)
        f.close()

    with open(args.train_config, 'r') as f:
        data_cfg = json.load(f)
        f.close()

    with open(args.env_config, 'r') as f:
        env_cfg = json.load(f)
        f.close()

    #build log
    log_dir = os.path.join(data_cfg['log_dir'], model_cfg['model']['type'])

    if not os.path.isdir(log_dir):
        os.mkdir(log_dir)

    log_dir = os.path.join(log_dir, 'train.log')
    logger = get_logger(log_dir=log_dir)

    #check configuration
    logger.info('Check the configuaration files.')
    check_cfg(model_cfg, data_cfg, env_cfg, mode=True)

    #set random option from seed
    if env_cfg['seed'] is not None:
        logger.info('Set the deterministic options from seed.')
        set_deterministic_option(env_cfg['seed'])

    #set world size
    logger.info('Set the world size.')
    set_world_size(env_cfg)

    #train
    train_module(model_cfg, data_cfg, env_cfg, logger)


if __name__ == '__main__':
    main()
