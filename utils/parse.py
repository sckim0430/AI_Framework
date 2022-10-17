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

def parse_loss_eval(cfg, mode='train'):
    """The operation for parse the loss and evaluation.

    Args:
        cfg (dict): The input config
        mode (str, optional): The parse evaluatio type. Defaults to 'train'.

    Raises:
        ValueError: The keyword should be in config file

    Returns:
        list[str]: The loss and evaluation name list.
    """

    if mode not in cfg['params']['evaluation']:
        raise ValueError("The '{}' should be in config file.".format(mode))

    ouptut_list = []

    for key in cfg['model']:
        if 'head' in key:
            for k in cfg['model'][key]:
                if 'loss' in k:
                    output_list.append(k)

    for k in cfg['params']['evaluation'][mode]:
        output_list.append(k)

    return output_list    

