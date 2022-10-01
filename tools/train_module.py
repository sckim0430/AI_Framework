"""The train module implementation.
"""
import os
import warnings

import torch
import torch.nn as nn
import torch.multiprocessing as mp
from torch.optim.lr_scheduler import StepLR

from builds.build import build_model, build_optimizer
from utils.environment import set_rank, init_process_group, set_device, set_model


def train_module(model_cfg, data_cfg, env_cfg, logger):
    """The operation for train module.

    Args:
        model_cfg (dict): The model config.
        data_cfg (dict): The data config.
        env_cfg (dict): The environment config.
        logger (logging.RootLogger): The logger.
    """

    if env_cfg['multiprocessing_distributed']:
        mp.sqawn(train_sub_module, nprocs=env_cfg['ngpus_per_node'], args=(
            model_cfg, data_cfg, env_cfg, logger))
    else:
        train_sub_module(env_cfg['gpu_id'], model_cfg,
                         data_cfg, env_cfg, logger)


def train_sub_module(gpu_id, model_cfg, data_cfg, env_cfg, logger):
    """The operation for sub train module.

    Args:
        gpu_id (int): The gpu id.
        model_cfg (dict): The model config.
        data_cfg (dict): The data config.
        env_cfg (dict): The environment config.
        logger (logging.RootLogger): The logger.
    """
    #set the gpu paramters
    logger.info('Set the gpu parameters.')
    env_cfg.update({'gpu_id': gpu_id})
    select_gpu = True if env_cfg['gpu_id'] is not None else False
    is_cuda = torch.cuda.is_available()
    device = set_device(env_cfg['gpu_id'])

    #distribution option
    if env_cfg['distributed']:
        logger.info('Set the rank.')
        set_rank(env_cfg)

        logger.info('Initalize the process group.')
        init_process_group(env_cfg)

        if is_cuda and select_gpu:
            batch_size = int(data_cfg['batch_size']/env_cfg['ngpus_per_node'])
            workers = int(
                (env_cfg['workers']+env_cfg['ngpus_per_node']-1)/env_cfg['ngpus_per_node'])

            logger.info('Convert the batch size {} -> {} and workers {} -> {} by multi processing.'.format(
                data_cfg['batch_size'], batch_size, env_cfg['workers'], workers))
            data_cfg.update({'batch_size': batch_size})
            env_cfg.update({'workers': workers})

    #build model
    logger.info('Build the model.')
    model = build_model(model_cfg['model'], logger)

    #set model
    logger.info('Set the model.')
    model = set_model(model, device, select_gpu=select_gpu,
                      distributed=env_cfg['distributed'])

    #build optimizer & scheduler
    logger.info('Build the optimizer and learning rate scheduler.')
    optimizer = build_optimizer(model.parameters(), model_cfg['optimizer'])
    #learninig rate will decayed by gamma every step_size epochs
    scheduler = StepLR(optimizer, step_size=30, gamma=0.1)

    #initalize the best evaluation scores
    best_eval = {k: 0 for k in model_cfg['params']
                 ['evaluation']['validation'].keys()}

    #load the resume model weight
    if data_cfg['resume'] is not None:
        if os.path.isfile(data_cfg['resume']):
            logger.info('Load the resume checkpoint : {}.'.format(
                data_cfg['resume']))

            checkpoint = torch.load(data_cfg['resume'], map_location=device)

            #update start epoch
            data_cfg.upate({'start_epoch': checkpoint['epoch']})

            #update best evaluation scores
            for k in checkpoint:
                if k not in ('epoch', 'architecture', 'model', 'optimizer', 'scheduler') and k not in best_eval.keys():
                    logger.warning('There is no evaluation : {}'.format(k))
                    continue

                best_eval.update({k: checkpoint[k].to()})

        else:
            logger.warning(
                'The wrong resume checkpoint : {}'.format(data_cfg['resume']))
