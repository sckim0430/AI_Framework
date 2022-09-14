"""Build Model from Config Options
"""
from models.type import *
from models.module import *
from utils.parse import parse_type


def build_model(model_cfg):
    """Build Model Class from Model Configuration

    Args:
        model_cfg (dict): Model Options
    """
    #parse model config
    model_type, model_params = parse_type(model_cfg)

    #build models
    if 'backbone' in model_params:
        model_params['backbone'] = build_sub_model(model_params['backbone'])

    if 'cls_head' in model_params:
        model_params['cls_head'] = build_sub_model(model_params['cls_head'])

    return eval(model_type)(**model_params)


def build_sub_model(cfg):
    """Build Sub Model Class

    Args:
        cfg (dict): Sub Model Config

    Returns:
        class: Sub Model Class Object
    """
    #parse type from config
    type, params = parse_type(cfg)

    return eval(type)(**params)
