"""The train module implementation.
"""
import os
import warnings

import torch
import torch.nn as nn
import torch.multiprocessing as mp
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from builds.build import build_model, build_optimizer, build_pipeline, build_dataset
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
    best_evaluation = {k: 0 for k in model_cfg['params']
                       ['evaluation']['validation'].keys()}

    #load the resume model weight
    if data_cfg['resume'] is not None:
        if os.path.isfile(data_cfg['resume']):
            logger.info('Load the resume checkpoint : {}'.format(
                data_cfg['resume']))

            checkpoint = torch.load(data_cfg['resume'], map_location=device)

            if model_cfg['model'] == checkpoint['architecture']:
                #update start epoch
                data_cfg.upate({'start_epoch': checkpoint['epoch']})

                #update best evaluation scores
                for k in checkpoint['best_evaluation']:
                    if k in best_evaluation:
                        best_evaluation.update(
                            {k: checkpoint['best_evaluation'][k].to(device)})
                    else:
                        warnings.warn(
                            'There is not best evaluation key : {}'.format(k))

                #load model, optimizer, scheduler state dict
                model.load_state_dict(checkpoint['model'])
                optimizer.load_state_dict(checkpoint['optimizer'])
                scheduler.load_state_dict(checkpoint['scheduler'])
            else:
                logger.warning(
                    'The resume checkpoint architecture does not match model config architecture. The resume checkpoint can not be loaded.')
        else:
            logger.warning(
                'The resume checkpoint have wrong path with {}. The resume checkpoint can not be loaded'.format(data_cfg['resume']))

    #generate the dataset
    if data_cfg['dummy']:
        logger.info('Generate the dummy data.')

        train_dataset = datasets.FakeData(
            1000000, (3, 224, 224), 1000, transforms.ToTensor())
        val_dataset = datasets.FakeData(
            50000, (3, 224, 224), 1000, transforms.ToTensor())
    else:
        logger.info('Generate the train/validate data.')

        train_pipeline = build_pipeline(model_cfg['pipeline'], mode="train")
        val_pipeline = build_pipeline(model_cfg['pipeline'], mode="validation")

        train_dataset = build_dataset(
            dataset=data_cfg['dataset'], root=data_cfg['train_dir'], transform=train_pipeline, split='train')
        val_dataset = build_dataset(
            data_cfg['dataset'], root=data_cfg['val_dir'], transforms=val_pipeline, split='val')

    #generate the sampler
    if env_cfg['distributed']:
        train_sampler = DistributedSampler(train_dataset)
        val_sampler = DistributedSampler(
            val_dataset, shuffle=False, drop_last=True)
    else:
        train_sampler = None
        val_sample = None

    #load the dataset loader
    train_loader = DataLoader(dataset=train_dataset, batch_size=data_cfg['batch_size'], shuffle=(
        train_sampler is None), num_workers=env_cfg['workers'], pin_memory=True, sampler=train_sampler)
    val_loader = DataLoader(dataset=val_dataset, batch_size=data_cfg['batch_size'],
                            shuffle=False, num_workers=env_cfg['workers'], pin_memory=True, sampler=val_sampler)

    for epoch in range(data_cfg['start_epoch'], data_cfg['epochs']):
        if env_cfg['distributed']:
            train_sampler.set_epoch(epoch)

        