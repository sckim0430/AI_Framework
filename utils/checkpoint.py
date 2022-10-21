"""The checkpoint operator implementation.
"""
import os
import shutil
from collections import OrderedDict
import torch


def load_pretrained_checkpoint(model, filename):
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
            src_state.update({k: v})

    model.load_state_dict(src_state)


def save_checkpoint(state, is_best, directory=None, file_name=None):
    """The operation for save checkpoint.

    Args:
        state (dict): The model state.
        is_best (defaultdict[bool]): The option for model have best evaluation result.
        directory (str, optional): The model directory. Defaults to None.
        file_name (str, optional): The model file name. Defaults to None.
    """
    if file_name is None:
        file_name = '{}_checkpoint.pth.tar'.format(state['epoch'])

    if directory is not None:
        if not os.path.isdir(directory):
            os.mkdir(directory)
        file_name = os.path.join(directory, file_name)

    torch.save(state, file_name)

    for k, v in is_best.items():
        if v:
            file_best_name = 'best_{}.pth.tar'.format(k)

            if directory is not None:
                file_best_name = os.path.join(directory, file_best_name)

            shutil.copyfile(file_name, file_best_name)
