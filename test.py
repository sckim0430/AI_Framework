"""The test implementation.
"""
import os
import json
import argparse

from utils.log import get_logger
from utils.check import check_cfg
from utils.environment import set_deterministic_option, set_world_size, set_workers, set_distributed
from tools.test_module import test_module


def parse_args():
    """The operation for parse arguments.

    Returns:
        argparse.Namespace : The arguments.

    Notice:
        We only use json format config file.
    """

    parser = argparse.ArgumentParser(description='Pytorch Imagenet Train')
    parser.add_argument(
        '--model_config_dir', default='/workspace/Benchmark/configs/classification/alexnet/alexnet_model.json', help='model config path')
    parser.add_argument(
        '--data_config_dir', default='/workspace/Benchmark/configs/classification/alexnet/alexnet_data.json', help='data config path')
    parser.add_argument(
        '--env_config_dir', default='/workspace/Benchmark/configs/env_config.json', help='environment config path')
    args = parser.parse_args()

    return args


def main():
    """The operation for main.
    """
    # load config
    args = parse_args()

    with open(args.model_config_dir, 'r') as f:
        model_cfg = json.load(f)
        f.close()

    with open(args.data_config_dir, 'r') as f:
        data_cfg = json.load(f)
        f.close()

    with open(args.env_config_dir, 'r') as f:
        env_cfg = json.load(f)
        f.close()

    # build log
    if data_cfg['log_dir'] is not None:
        log_dir = data_cfg['log_dir']
    else:
        log_dir = 'log'

    if not os.path.isdir(log_dir):
        os.mkdir(log_dir)

    log_dir = os.path.join(log_dir, model_cfg['model']['type'])

    if not os.path.isdir(log_dir):
        os.mkdir(log_dir)

    log_dir = os.path.join(log_dir, 'test.log')
    log_option = {"log_level": 1, "stream_level": 1,
                  "file_level": 2, "log_dir": log_dir, "distributed": False}
    logger = get_logger(**log_option)

    # check configuration
    logger.info('Check the configuaration files.')
    check_cfg(model_cfg, data_cfg, env_cfg, mode=False)

    # set random option from seed
    if env_cfg['seed'] is not None:
        logger.info('Set the deterministic options from seed.')
        set_deterministic_option(env_cfg['seed'])

    # set distirbution
    logger.info('Set the distirbution.')
    set_distributed(env_cfg)
    log_option.update({"distributed": env_cfg['distributed']})
    # set world size
    logger.info('Set the world size.')
    set_world_size(env_cfg)
    # set workers
    logger.info('Set the workers.')
    set_workers(env_cfg)

    # test
    test_module(model_cfg, data_cfg, env_cfg, log_option)


if __name__ == '__main__':
    main()
