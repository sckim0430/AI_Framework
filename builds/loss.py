"""Build Loss from Config Options
"""
from models.module.loss import *
from utils.parse import parse_type

def build_loss(cfg):
    """Build Loss Function from Config

    Args:
        cfg (dict): loss function options

    Returns:
        class: loss function 
    """
    type, params = parse_type(cfg)

    eval(type)(**params)