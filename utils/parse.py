"""Parse the utils.
"""

def parse_type(cfg):
    """Parse the type keyword from config file.

    Args:
        cfg (dict): The configuration.

    Returns:
        str, dict: The type and parameters.
    """
    assert "type" in cfg, "The config file should have 'type' key."

    parsed_cfg = cfg.copy()

    type = parsed_cfg['type']
    del parsed_cfg['type']
    params = parsed_cfg

    return type, params