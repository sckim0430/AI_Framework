"""Build from Config Options
"""
from turtle import back
from models.type import *
from models.module import *

def build_model(model_cfg):
    """Build Model Class from Model Configuration

    Args:
        model_cfg (dict): Model Options
    """
    model_type = model_cfg['type']
    backbone_cfg = model_cfg['backbone']
    cls_head_cfg = model_cfg['cls_head']
    init_weight = model_cfg['init_weight']

    #parse backbone config
    backbone_type = backbone_cfg['type']
    # backbone_

    #parse cls head config

    #init weight

    exec(model_type+'({},{},{})'.format())
    