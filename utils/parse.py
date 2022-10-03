"""The parse utils implementation.
"""


def parse_type(cfg):
    """The operation for parse the type keyword from config.

    Args:
        cfg (dict): The input config.

    Raises:
        ValueError: The config file should have 'type' key.

    Returns:
        str, dict: The type and parameters.
    """
    if "type" not in cfg:
        raise ValueError("The config file should have 'type' key.")

    parsed_cfg = cfg.copy()

    type = parsed_cfg['type']
    del parsed_cfg['type']
    params = parsed_cfg

    return type, params
