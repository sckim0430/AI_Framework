"""The train module implementation.
"""
import os
import warnings
from time import time

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from builds.build import build_model, build_optimizer, build_pipeline, build_dataset, build_param
from utils.environment import set_rank, init_process_group, set_device, set_model
from utils.parse import parse_loss_eval
from utils.display import display
from utils.AverageMeter import AverageMeter, MetricMeter

def train_module(model_cfg, data_cfg, env_cfg, logger):
    """The operation for train module.

    Args:
        model_cfg (dict): The model config.
        data_cfg (dict): The data config.
        env_cfg (dict): The environment config.
        logger (logging.RootLogger): The logger.
    """
    #run the train module
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
    if is_cuda and env_cfg['distributed']:
        logger.info('Set the rank.')
        set_rank(env_cfg)

        logger.info('Initalize the distributed process group.')
        init_process_group(env_cfg['dist_url'],env_cfg['dist_backend'],env_cfg['world_size'],env_cfg['rank'])

        if select_gpu:
            batch_size = int(data_cfg['batch_size']/env_cfg['ngpus_per_node'])
            workers = int(
                (env_cfg['workers']+env_cfg['ngpus_per_node']-1)/env_cfg['ngpus_per_node'])

            logger.info('Convert the batch size {} -> {} and workers {} -> {} by single gpu and multi processing.'.format(
                data_cfg['batch_size'], batch_size, env_cfg['workers'], workers))
            data_cfg.update({'batch_size': batch_size})
            env_cfg.update({'workers': workers})

    #build model
    logger.info('Build the model.')
    model = build_model(model_cfg['model'], logger)

    #set model
    logger.info('Set the model.')
    model = set_model(model, device, select_gpu,
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
        logger.info('Set the train/validation sampler.')
        train_sampler = DistributedSampler(train_dataset)
        val_sampler = DistributedSampler(
            val_dataset, shuffle=False, drop_last=True)
    else:
        train_sampler = None
        val_sample = None

    #load the dataset loader
    logger.info('Set the train/validation data loader.')
    train_loader = DataLoader(dataset=train_dataset, batch_size=data_cfg['batch_size'], shuffle=(
        train_sampler is None), num_workers=env_cfg['workers'], pin_memory=True, sampler=train_sampler)
    val_loader = DataLoader(dataset=val_dataset, batch_size=data_cfg['batch_size'],
                            shuffle=False, num_workers=env_cfg['workers'], pin_memory=True, sampler=val_sampler)

    #get train/validation params
    train_freq = data_cfg['train_freq'] if 'train_freq' in data_cfg else None
    val_freq = data_cfg['val_freq'] if 'val_freq' in data_cfg else None

    validate_mode = val_freq is not None

    train_params = build_param(model_cfg['params'],mode='train')

    if validate_mode:
        val_params = build_param(model_cfg['params'],mode='validation')

    logger.info('Train start.')
    for epoch in range(data_cfg['start_epoch'], data_cfg['epochs']):
        if env_cfg['distributed']:
            train_sampler.set_epoch(epoch)

        if train_freq is None:
            train(train_loader, model, train_params, optimizer, epoch, device)
        else:
            train(train_loader, model, train_params, optimizer, epoch, device, train_freq)

        try:
            if validate_mode and epoch%val_freq==0:
                validate(val_loader, model, val_params, epoch, device, best_evaluation,env_cfg['distributed'],env_cfg['world_size'])
        except ZeroDivisionError:
            warnings.warn('The val_freq value should not be zero. Set the val_freq to 5.')
            val_freq=5

        scheduler.step()

        if not env_cfg['distributed'] or (env_cfg['distributed'] and (select_gpu or env_cfg['rank']%env_cfg['ngpus_per_node']==0)):
            logger.info('Save checkpoint..{} epoch.'.format(epoch))
            pass

def train(data_loader, model, params, optimizer, epoch, device, train_freq=5):
    """The operation for train every epoch call.

    Args:
        data_loader (torch.utils.data.DataLoader): The train data loader.
        model (nn.Module): The model.
        params (dict): The train parameters.
        optimizer (Optimizer): The optimizer.
        epoch (int): The epoch.
        device (torch.device): The device.
        train_freq (int): The train frequent. Defaults to 5.
    """
    mode = 'train'
    data_time = AverageMeter('data load time',prefix=mode)
    batch_time = AverageMeter('batch inference time',prefix=mode)
    metrics = MetricMeter(prefix=mode)

    model.train()
    end = time()
    #train loop
    for i, (images,targets) in enumerate(data_loader):
        #data load time update
        data_time.update(time()-end)

        images.to(device,non_blocking=True)
        targets.to(device,non_blocking=True)
        
        #get output[losses, evaluations, .., etc.]
        output = model(images,targets,return_loss=True,**params)

        #output update
        metrics.update(output)

        #compute gradient and optimizer step
        optimizer.zero_grad()

        for k in output:
            if 'loss' in k:
                output[k].backward()

        optimizer.step()
        
        #elapsed time update
        batch_time.update(time()-end)
        end = time()

        try:
            #display
            if i%train_freq==0:
                display(epoch, len(data_loader), i+1, metrics, data_time, batch_time)
        except ZeroDivisionError:
            warnings.warn('The train_freq value should not be zero. Set the train_freq to 5.')
            train_freq=5


def validate(data_loader, model, params, epoch, device, best_evaluation,distributed,world_size):
    """The operation for validation every epoch call.

    Args:
        data_loader (torch.utils.data.DataLoader): The validation data loader.
        model (nn.Module): The model.
        params (dict): The validation parameters.
        epoch (int): The epoch.
        device (torch.device): The torch device.
        best_evaluation (dict): The best evaluation results on the validation dataset.
        distributed (bool): The option for distribution.
        world_size (int): The world size..
    """
    def run_validate(loader):
        with torch.no_grad():
            end = time()

            for (images, targets) in loader:
                #data load time update
                data_time.update(time()-end)
                
                images.to(device,non_blocking=True)
                targets.to(device.non_blocking=True)

                #get validation params and output[losses, evaluations, .., etc.]
                output = model(images,targets,return_loss=True,**params)

                #output update
                metrics.update(output)

                #elapsed time update
                batch_time.update(time()-end)
                end = time()

    mode = 'validation'
    data_time = AverageMeter('data load time',prefix=mode)
    batch_time = AverageMeter('batch inference time',prefix=mode)
    metrics = MetricMeter(prefix=mode)

    model.eval()
    run_validate(data_loader)

    if distributed:
        data_time.all_reduce(device=device)
        batch_time.all_reduce(device=device)
        metrics.all_reduce(device=device)

        #aux validation set processing
        if len(data_loader.sampler)*world_size<len(data_loader.dataset):
            aux_val_dataset = Subset(data_loader.dataset,range(len(data_loader.sampler)*world_size,len(data_loader.dataset)))
            aux_val_loader = DataLoader(aux_val_dataset,batch_size=data_loader.batch_size/world_size,shuffle=False,num_workers=int((data_loader.workers+world_size-1)/world_size),pin_memory=True)
            run_validate(aux_val_loader)

    #best evaluation initialization
    for k,v in best_evaluation.items():
        best_evaluation.update({k:max(v,metrics.meters[k].avg)})

    #display
    display(epoch,len(data_loader),len(data_loader),metrics, data_time, batch_time)