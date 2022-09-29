"""The build implementation.
"""
from models.type import *
from models.module import *
from utils.parse import parse_type


def build(cfg, log_manager=None):
    """The operation for build.

    Args:
        cfg (dict): The input config.
        log_manager (builds.log.LogManager): The log manager. Defaults to None.

    Returns:
        class: The class object.
    """
    #parse type from config
    type, params = parse_type(cfg)
    params.update({'log_manager': log_manager})

    return eval(type)(**params)


def build_model(cfg, log_manager=None):
    """The operation for build model.

    Args:
        cfg (dict): The input config.
        log_manager (builds.log.LogManager): The log manager. Defaults to None.
    Returns:
        class: The class object.
    """
    #parse model config
    type, params = parse_type(cfg)

    #build sub modules
    for k in params:
        params.update({k: build(params[k], log_manager)})

    params.update({'log_manager': log_manager})

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
    cfg_param.update({'evaluation': cfg_param['evaluation'][type]})

    return cfg_param
