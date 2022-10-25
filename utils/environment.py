"""The setting environment implementation.
"""
import os
import warnings
import random
import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.backends.cudnn as cudnn


def set_deterministic_option(seed):
    """The operation for set random option from seed.

    Args:
        seed (int): The seed.
    """
    assert seed is not None, "When setting the deterministic option, the seed should not None."

    random.seed(seed)
    np.random.seed(seed)
    torch.matmul_seed(seed)
    cudnn.deterministic = True

    warnings.warn('You have chosen to seed training. '
                  'This will turn on the CUDNN deterministic setting, '
                  'which can slow down your training considerably! '
                  'You may see unexpected behavior when restarting '
                  'from checkpoints.')

##############################################################################################################################
# GPU CASE

# 1. cpu
# 2. single gpu, single node, single process(this, total)
# 3. single gpu, multi node, single process(this) / multi process(total)
# 4. multi gpu, single node, single process(this, total)
# 5. multi gpu, single node, multi process(this, total)
# 6. multi gpu, multi node, multi process(this, total)
# 7. multi gpu, multi node, single process(this) / multi process(total)
##############################################################################################################################
# dist_url

# 1. as in the case of 2 and 4, distributed training is not performed in a single process, so world_size and rank are ignored.
# 2. if dist_url is specified as env://, it is assumed that world_size, rank, master_port, and master_ip are specified in environment variables.
# 3. if world_size and rank information is not included in dist_url, world_size and rank must be initialized.
# 4. if dist_url includes world_size and rank information, it is not necessary to initialize world_size and rank on environment variable.this is not used now.
# 5. if dist_url is not used, the store option must be used, and world_size and rank must be initialized on environment variable. this is not used now.
##############################################################################################################################


def set_distributed(env_cfg):
    """The operation for distribution.

    Args:
        env_cfg (dict): The environment config.
    """
    env_cfg.update(
        {'distributed': env_cfg['world_size'] > 1 or env_cfg['multiprocessing_distributed']})


def set_workers(env_cfg):
    """The operation for set the workers.

    Args:
        env_cfg (dict): The environment config.
    """
    if env_cfg['distributed']:
        env_cfg.update({'workers': int((
            env_cfg['workers']+env_cfg['ngpus_per_node']-1)/env_cfg['ngpus_per_node'])})


def set_world_size(env_cfg):
    """The operation for set the world size.

    Args:
        env_cfg (dict): The environment config.
    """
    # ngpus_per_node : gpu number per node
    # we assign 1 process per gpu.
    env_cfg.update({'ngpus_per_node': torch.cuda.device_count()})

    # the world size means node number to process number,
    # so, in case with 5 and 6, we redefine world size = ngpus_per_node * world size.
    if env_cfg['multiprocessing_distributed']:
        env_cfg.update(
            {'world_size': env_cfg['ngpus_per_node'] * env_cfg['world_size']})


def set_rank(env_cfg, gpu_id):
    """The operation for set rank.

    Args:
        env_cfg (dict): The environment config.
        gpu_id (int): The local rank.
    """
    # rank means the priority of the current node among all nodes,
    # so, in case with 5 and 6, we redefine rank = rank * ngpus_per_node + gpu_id.
    # finally, it is changed from the priority of the current node to the priority of the process.
    if env_cfg['multiprocessing_distributed']:
        env_cfg.update(
            {'rank': env_cfg['rank']*env_cfg['ngpus_per_node']+gpu_id})


def init_process_group(dist_url, dist_backend, world_size, rank):
    """The operation for initalize process group.

    Args:
        env_cfg (dict): The environment config.
    """
    dist.init_process_group(init_method=dist_url, backend=dist_backend,
                            world_size=world_size, rank=rank)


def set_device(local_rank=None):
    """The operation for set torch device.
    Args:
        local_rank (int): The local rank. Defaults to None.
    Returns:
        torch.device: The torch device.
    """
    device = None

    if torch.cuda.is_available():
        if local_rank is not None:
            device = torch.device('cuda:{}'.format(local_rank))
        else:
            device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')

    return device


def set_model(model, device, distributed=False):
    """The operation for set model's distribution mode.
    Args:
        model (nn.Module): The model.
        device (torch.device): The torch device.
        distributed (bool, optional): The option for distributed. Defaults to False.
    Raises:
        ValueError: If distributed gpu option is true, the gpu device should cuda.
    Returns:
        nn.Module: The model.
    """
    is_cuda = torch.cuda.is_available()

    if distributed:
        if is_cuda:
            model.to(device)
            model = nn.parallel.DistributedDataParallel(
                model, device_ids=[device], output_device=[device])
        else:
            raise ValueError(
                'If in cpu or mps mode, distributed option should be False.')
    else:
        model = model.to(device)

        if is_cuda and torch.cuda.device_count() > 1:
            model = nn.parallel.DataParallel(model)

    return model
