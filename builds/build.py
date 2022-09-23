"""Build the model from config option.
"""
from models.type import *
from models.module import *
from utils.parse import parse_type


def build(cfg):
    """Build the class.

    Args:
        cfg (dict): Sub config.

    Returns:
        class: Sub class object.
    """
    #parse type from config
    type, params = parse_type(cfg)

    return eval(type)(**params)


def build_model(cfg):
    """Build the model class from config option.

    Args:
        cfg (dict): Model options.

    Returns:
        class: Model class object.
    """
    #parse model config
    type, params = parse_type(cfg)

    #build sub modules
    for k in params:
        params[k] = build(params[k])

    return eval(type)(**params)


def build_param(cfg, type='train'):
    """Build the param from config with type.

    Args:
        cfg (_type_): Parameter options.
        type (str, optional): Parameter type. Defaults to 'train'.

    Returns:
        dict: Built parameter options.
    """
    assert 'evaluation' in cfg, "The config file must have 'evaluation' key."
    assert type in cfg['evaluation'] and type in (
        'train', 'validation', 'test'), "The type should be in ('train', 'validation', 'test') and config."

    cfg_param = cfg.copy()
    cfg_param['evaluation'] = cfg_param['evaluation'][type]

    return cfg_param
