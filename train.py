"""Train Script For Classification Model
"""
import argparse
import json
from utils.set_env import set_deterministic_option, set_world_size
from tools.train_module import train_module


def parse_args():
    """Generate Arguments
    Returns:
        argparse.Namespace : arguments
    Notice:
        We Only Use Json Format Config File
    """

    parser = argparse.ArgumentParser(description='Pytorch Imagenet Train')
    parser.add_argument(
        '--model_config_dir', default='/home/sckim/AI_Framework/configs/classification/alexnet/alexnet.json', help='model config path')
    parser.add_argument(
        '--train_config_dir', default='/home/sckim/AI_Framework/configs/classification/train_config.json', help='train config path')
    parser.add_argument(
        '--env_config_dir', default='/home/sckim/AI_Framework/configs/env_config.json', help='environment config path')
    args = parser.parse_args()

    return args


def main():
    """Main Worker Controller
    """

    #load config
    args = parse_args()

    with open(args.model_config, 'r') as f:
        model_cfg = json.load(f)
        f.close()

    with open(args.train_config, 'r') as f:
        train_cfg = json.load(f)
        f.close()

    with open(args.env_config, 'r') as f:
        env_cfg = json.load(f)
        f.close()

    #set random option from seed
    if env_cfg['seed'] is not None:
        set_deterministic_option(env_cfg['seed'])

    #set world size
    set_world_size(env_cfg)

    #train
    train_module(model_cfg, train_cfg, env_cfg)


if __name__ == '__main__':
    main()
