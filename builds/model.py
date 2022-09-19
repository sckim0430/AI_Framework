"""Build Model from Config Options
"""
from models.type import *
from models.module import *
from utils.parse import parse_type


def build_model(model_cfg):
    """Build Model Class from Model Configuration

    Args:
        model_cfg (dict): model options
    """
    #parse model config
    type, params = parse_type(model_cfg)

    #build models
    if 'backbone' in params:
        # assert params['backbone']
        params['backbone'] = build_sub_model(params['backbone'])

    if 'cls_head' in params:
        params['cls_head'] = build_sub_model(params['cls_head'])

    return eval(type)(**params)


def build_sub_model(cfg):
    """Build Sub Model Class

    Args:
        cfg (dict): sub model config

    Returns:
        class: sub model class object
    """
    #parse type from config
    type, params = parse_type(cfg)

    return eval(type)(**params)
