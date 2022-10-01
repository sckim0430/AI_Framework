"""The build implementation.
"""
import torch.optim as optim

from models.type import *
from models.module import *
from utils.parse import parse_type


def build(cfg, logger=None):
    """The operation for build.

    Args:
        cfg (dict): The input config.
        logger (logging.RootLogger): The logger. Defaults to None.

    Returns:
        nn.Module: The sub model object.
    """
    #parse type from config
    type, params = parse_type(cfg)
    params.update({'logger': logger})

    return eval(type)(**params)


def build_model(cfg, logger=None):
    """The operation for build model.

    Args:
        cfg (dict): The input config.
        logger (logging.RootLogger): The logger. Defaults to None.
    Returns:
        nn.Module: The model object.
    """
    #parse model config
    type, params = parse_type(cfg)

    #build sub modules
    for k in params:
        params.update({k: build(params[k], logger)})

    params.update({'logger': logger})

    return eval(type)(**params)


def build_optimizer(params, cfg):
    """The operation for build optimizer.

    Args:
        params (generator): The model parameters.
        cfg (dict): The input config.

    Returns:
        torch.optim: The optimizer object.
    """
    #parse optimizer config
    type, params = parse_type(cfg)
    params.update({'params': params})

    #build optimizer
    return optim.eval(type)(**params)


def build_param(cfg, type='train'):
    """The operation for build parameter.

    Args:
        cfg (dict): The input config.
        type (str, optional): The parameter about type. Defaults to 'train'.

    Returns:
        dict: The output config.
    """
    assert type in cfg['evaluation'] and type in (
        'train', 'validation', 'test'), "The type should be in ('train', 'validation', 'test') and config."

    cfg_param = cfg.copy()
    cfg_param.update({'evaluation': cfg_param['evaluation'][type]})

    return cfg_param
