"""Train Module
"""
import torch.multiprocessing as mp
from utils.set_env import set_rank, init_process_group
from builds.build import build_model

def train_module(model_cfg, data_cfg, env_cfg):
    """Main Module Train Function

    Args:
        model_cfg (dict): model options
        train_cfg (dict): train options
        env_cfg (dict): environment options
    """

    if env_cfg['multiprocessing_distributed']:
        mp.sqawn(train_sub_module, nprocs=env_cfg['ngpus_per_node'], args=(
            model_cfg, data_cfg, env_cfg))
    else:
        train_sub_module(env_cfg['gpu_id'], model_cfg, data_cfg, env_cfg)


def train_sub_module(gpu_id, model_cfg, data_cfg, env_cfg):
    """Sub Module Train Function

    Args:
        gpu_id (int, Optional): the gpu id
        model_cfg (dict): model options
        train_cfg (dict): train options
        env_cfg (dict): environment options
    """
    #gpu : 현재 프로세스에서 사용하는 gpu id
    #None인 경우에는 gpu를 사용할 수 있다면 전체 gpu 목록에 모델 weight를 할당한다. (model.cuda())
    #특정 gpu가 지정된 경우에는 해당 gpu에만 모델 weight를 할당한다. (torch.cuda.set_device(gpu_id), model.cuda(gpu_id))
    env_cfg['gpu_id'] = gpu_id

    #notice
    if env_cfg['gpu_id'] is None:
        print('Use All GPU Which You Can Use or CPU for Training in This Node with Single Processing')
    else:
        print('Use GPU : {} for Training in This Node with Multi Processing'.format(
            env_cfg['gpu_id']))

    #distribution option
    if env_cfg['distributed']:
        set_rank(env_cfg)
        init_process_group(env_cfg)

    #create model(build)
    model = build_model(model_cfg)
