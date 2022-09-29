"""The train module implementation.
"""
import os
import warnings

import torch
import torch.nn as nn
import torch.multiprocessing as mp
from torch.optim.lr_scheduler import StepLR

from builds.build import build_model, build_optimizer
from utils.set_env import set_rank, init_process_group


def train_module(model_cfg, data_cfg, env_cfg, log_manager):
    """The operation for train module.

    Args:
        model_cfg (dict): The model config.
        data_cfg (dict): The data config.
        env_cfg (dict): The environment config.
        log_manager (builds.log.LogManager): The log manager.
    """

    if env_cfg['multiprocessing_distributed']:
        mp.sqawn(train_sub_module, nprocs=env_cfg['ngpus_per_node'], args=(
            model_cfg, data_cfg, env_cfg, log_manager))
    else:
        train_sub_module(env_cfg['gpu_id'], model_cfg,
                         data_cfg, env_cfg, log_manager)


def train_sub_module(gpu_id, model_cfg, data_cfg, env_cfg, log_manager):
    """The operation for sub train module.

    Args:
        gpu_id (int): The gpu id.
        model_cfg (dict): The model config.
        data_cfg (dict): The data config.
        env_cfg (dict): The environment config.
        log_manager (builds.log.LogManager): The log manager.
    """
    #gpu : 현재 프로세스에서 사용하는 gpu id
    #None인 경우에는 gpu를 사용할 수 있다면 전체 gpu 목록에 모델 weight를 할당한다. (model.cuda())
    #특정 gpu가 지정된 경우에는 해당 gpu에만 모델 weight를 할당한다. (torch.cuda.set_device(gpu_id), model.cuda(gpu_id))
    env_cfg.update({'gpu_id': gpu_id})

    #distribution option
    if env_cfg['distributed']:
        log_manager.logger.info('Set the rank.')
        set_rank(env_cfg)

        log_manager.logger.info('Initalize the process group.')
        init_process_group(env_cfg)

    #create model(build)
    log_manager.logger.info('Build the model.')
    model = build_model(model_cfg['model'], log_manager)

    log_manager.logger.info('Set the distribution and gpu options.')
    if torch.cuda.is_available():
        warnings.warn(
            'You will use cpu and the training speed will very slow.')
    elif env_cfg['distributed']:
        if env_cfg['gpu_id'] is not None:
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs of the current node.
            print('This is the case with other nodes including the current node with multi gpus in multi processing. You will use {} gpu now.'.format(
                env_cfg['gpu_id']))
            torch.cuda.set_device(env_cfg['gpu_id'])
            model.cuda(env_cfg['gpu_id'])
            data_cfg.update(
                {'batch_size': int(data_cfg['batch_size']/env_cfg['ngpus_per_node'])})
            env_cfg.update({'workers': int(
                (env_cfg['workers']+env_cfg['ngpus_per_node']-1)/env_cfg['ngpus_per_node'])})
            model = nn.parallel.DistributedDataParallel(
                model, device_ids=[env_cfg['gpu_id']])
        else:
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            print('This is the case with other nodes including the current node with a single gpu in multi processing. You will use all available gpus.')
            model.cuda()
            nn.parallel.DistributedDataParallel(model)
    elif env_cfg['gpu_id'] is not None:
        print('This is the case with single node with single gpu in single processing. You will use {} gpu now.'.format(
            env_cfg['gpu_id']))
        torch.cuda.set_device(env_cfg['gpu_id'])
        model = model.cuda(env_cfg['gpu_id'])
    else:
        print('This is the case with single node with multi gpus in single processing, You will use all available gpus.')
        model.cuda()
        model = nn.parallel.DataParallel(model)

    #build optimizer & scheduler
    log_manager.logger.info('Build the optimizer and learning rate scheduler.')
    optimizer = build_optimizer(model.parameters(), model_cfg['optimizer'])
    # learninig rate will decayed by gamma every step_size epochs
    scheduler = StepLR(optimizer, step_size=30, gamma=0.1)

    if data_cfg['resume'] is not None:
        if os.path.isfile(data_cfg['resume']):
            
            pass
        else:
            warnings.warn('There is no checkpoint : {}'.format(data_cfg['resume']))