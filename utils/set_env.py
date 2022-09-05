import os
import random
import numpy as np
import torch
import torch.distributed as dist
import torch.backends.cudnn as cudnn

import warnings


def set_deterministic_option(seed):
    """set random option from seed

    Args:
        seed (int): the seed.
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
#DISTRIBUTED
##############################################################################################################################
#여섯 가지의 경우의 수
#1. cpu
#2. single gpu, single process, single node
#3. single gpu, multi process, multi node
#4. multi gpu, single process, single node
#5. multi gpu, multi process, single node
#6. multi gpu, multi process, multi node
##############################################################################################################################
#dist_url : 프로세스 그룹을 초기화하는 방법을 명시한 url을 의미한다.
#1. (2,4)와 같은 single process에서는 distributed training을 하지 않기 떄문에, world_size와 rank를 무시한다.
#2. dist_url이 env://로 지정된 경우에는 환경 변수에 world_size, rank, master_port, master_ip가 지정된 것으로 간주한다.
#3. dist_url에 world_size, rank 정보가 포함되어있지 않은 경우에는 world_size, rank를 초기화 시켜야한다.
#4. dist_url에 world_size, rank 정보가 포함되어있는 경우에는 world_size, rank를 초기화 시키기 않아도 된다. => 이 코드에선 사용 x
#5. dist_url을 사용하지 않는 경우에는 store 옵션을 사용해야되며, world_size, rank는 필수로 초기화 시켜야한다. => 이 코드에선 사용 x
##############################################################################################################################


def set_world_size(env_cfg):
    """Set World Size
    Args:
        env_cfg (dict): environment options
    """

    if env_cfg['dist_url'] == 'env://' and env_cfg['world_size'] == -1:
        env_cfg['world_size'] = int(os.environ['WORLD_SIZE'])

    #multiprocessing_distributed 옵션은 이 노드에서 멀티 프로세스 동작시킬지에 대한 옵션(5,6)
    #world_size>1인 경우는 멀티 노드인 경우를 의미한다.(3)
    #즉, distributed는 멀티 프로세스 옵션
    env_cfg['distributed'] = env_cfg['world_size'] > 1 or env_cfg['multiprocessing_distributed']

    #노드 당 gpu 수를 의미
    #gpu 1개 당 1개의 프로세스를 할당
    env_cfg['ngpus_per_node'] = torch.cuda.device_count()

    #world_size는 노드 수에서 프로세스의 수를 의미하게 되어, (5,6)의 경우에는
    #world_size = ngpus_per_node * world_size로 재정의한다. (rank도 재정의)
    if env_cfg['multiprocessing_distributed']:
        env_cfg['world_size'] = env_cfg['ngpus_per_node'] * \
            env_cfg['world_size']


def set_rank(env_cfg):
    """Set Rank

    Args:
        env_cfg (dict): environment options
    """
    assert env_cfg['distributed']

    #dist_url이 env://인 경우에는 환경 변수의 값을 참조하여 구한다.
    if env_cfg['dist_url'] == 'env://' and env_cfg['rank'] == -1:
        env_cfg['rank'] = int(os.environ['RANK'])

    #gpu_cfg['rank']는 전체 노드중에서 현재 노드의 우선순위를 의미하므로,
    #현재 노드의 우선순위 * 각 노드당 gpu 수 + 현재 gpu id(순위)로 갱신한다.
    #현재 노드의 우선순위에서 프로세스의 순위로 변경된다.
    if env_cfg['multiprocessing_distributed']:
        env_cfg['rank'] = env_cfg['rank'] * \
            env_cfg['ngpus_per_node'] + env_cfg['gpu_id']


def init_process_group(env_cfg):
    """Initalize Process Group

    Args:
        env_cfg (dict): environment options
    """
    assert env_cfg['distributed']

    dist.init_process_group(init_method=env_cfg['dist_url'], backend=env_cfg['dist_backend'],
                            world_size=env_cfg['world_size'], rank=env_cfg['rank'])
