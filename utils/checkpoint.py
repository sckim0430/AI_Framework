"""The checkpoint operator implementation.
"""
from collections import OrderedDict
import torch


def load_checkpoint(model, filename):
    """The operation for load checkpoint.

    Args:
        model (nn.Module): The model class.
        filename (str): The pretrained model path.
    """
    src_state = model.state_dict()

    dst_checkpoint = torch.load(filename)
    dst_state = dst_checkpoint['model']

    for k, v in dst_state.items():
        if k in src_state:
            src_state[k] = v

    model.load_state_dict(src_state)
