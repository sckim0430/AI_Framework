"""The setting environment implementation.
"""
import os
import random
import numpy as np
import torch
import torch.distributed as dist
import torch.backends.cudnn as cudnn

import warnings


def set_deterministic_option(seed):
    """The operation for set random option from seed.

    Args:
        seed (int): The seed.
    """
    assert seed is not None

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
#GPU CASE

#1. cpu
#2. single gpu, single node, single process
#3. single gpu, multi node, multi process
#4. multi gpu, single node, single process
#5. multi gpu, single node, multi process
#6. multi gpu, multi node, multi process
##############################################################################################################################
#dist_url

#1. as in the case of 2 and 4, distributed training is not performed in a single process, so world_size and rank are ignored.
#2. if dist_url is specified as env://, it is assumed that world_size, rank, master_port, and master_ip are specified in environment variables.
#3. if world_size and rank information is not included in dist_url, world_size and rank must be initialized.
#4. if dist_url includes world_size and rank information, it is not necessary to initialize world_size and rank on environment variable.this is not used now.
#5. if dist_url is not used, the store option must be used, and world_size and rank must be initialized on environment variable. this is not used now.
##############################################################################################################################


def set_world_size(env_cfg):
    """The operation for set the world size.

    Args:
        env_cfg (dict): The environment config.
    """

    if env_cfg['dist_url'] == 'env://' and env_cfg['world_size'] == -1:
        env_cfg.update({'world_size': int(os.environ['WORLD_SIZE'])})

    #multiprocessing_distributed is option for multi processing in this node case with 5 and 6.
    #when world_size>1, then multi node case with 3.
    #distributed is option for total multi process.
    env_cfg.update(
        {'distributed': env_cfg['world_size'] > 1 or env_cfg['multiprocessing_distributed']})

    #ngpus_per_node : gpu number per node
    #we assign 1 process per gpu.
    env_cfg.update({'ngpus_per_node': torch.cuda.device_count()})

    #the world size means node number to process number,
    #so, in case with 5 and 6, we redefine world size = ngpus_per_node * world size.
    if env_cfg['multiprocessing_distributed']:
        env_cfg.update(
            {'world_size': env_cfg['ngpus_per_node'] * env_cfg['world_size']})


def set_rank(env_cfg):
    """The operation for set rank.

    Args:
        env_cfg (dict): The environment config.
    """
    #when dist_url == env://, we refer to environment variable.
    if env_cfg['dist_url'] == 'env://' and env_cfg['rank'] == -1:
        env_cfg.update({'rank': int(os.environ['RANK'])})

    #rank means the priority of the current node among all nodes,
    #so, in case with 5 and 6, we redefine rank = rank * ngpus_per_node + gpu_id.
    #finally, it is changed from the priority of the current node to the priority of the process.
    if env_cfg['multiprocessing_distributed']:
        env_cfg.update(
            {'rank': env_cfg['rank']*env_cfg['ngpus_per_node']+env_cfg['gpu_id']})


def init_process_group(env_cfg):
    """The operation for initalize process group.

    Args:
        env_cfg (dict): The environment config.
    """
    dist.init_process_group(init_method=env_cfg['dist_url'], backend=env_cfg['dist_backend'],
                            world_size=env_cfg['world_size'], rank=env_cfg['rank'])
