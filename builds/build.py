"""The build implementation.
"""
from models.type import *
from models.module import *
from utils.parse import parse_type


def build(cfg):
    """The operation for build.

    Args:
        cfg (dict): The input config.

    Returns:
        class: The class object.
    """
    #parse type from config
    type, params = parse_type(cfg)

    return eval(type)(**params)


def build_model(cfg):
    """The operation for build model.

    Args:
        cfg (dict): The input config.

    Returns:
        class: The class object.
    """
    #parse model config
    type, params = parse_type(cfg)

    #build sub modules
    for k in params:
        params[k] = build(params[k])

    return eval(type)(**params)


def build_param(cfg, type='train'):
    """The operation for build parameter.

    Args:
        cfg (dict): The input config.
        type (str, optional): The parameter about type. Defaults to 'train'.

    Returns:
        dict: The output config.
    """
    assert 'evaluation' in cfg, "The config file must have 'evaluation' key."
    assert type in cfg['evaluation'] and type in (
        'train', 'validation', 'test'), "The type should be in ('train', 'validation', 'test') and config."

    cfg_param = cfg.copy()
    cfg_param['evaluation'] = cfg_param['evaluation'][type]

    return cfg_param
