"""Parsing Utils
"""

def parse_type(cfg):
    """Parse Type Keyword from Config

    Args:
        cfg (dict): model configuration

    Returns:
        str, dict: model class name, model class parameters
    """
    parsed_cfg = cfg.copy()

    assert 'type' in parsed_cfg, "keyword 'type' not in cfg"

    type = parsed_cfg['type']
    del parsed_cfg['type']
    params = parsed_cfg

    return type, params