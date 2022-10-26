"""The test module implementation.
"""
import os
import warnings
from time import time

import torch
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from builds.build import build_model, build_optimizer, build_scheduler, build_pipeline, build_dataset, build_param
from utils.environment import set_rank, init_process_group, set_device, set_model
from utils.display import display_test
from utils.AverageMeter import AverageMeter, MetricMeter
from utils.log import get_logger

warnings.filterwarnings(action='ignore')


def test_module(model_cfg, data_cfg, env_cfg, log_option):
    """The operation for test module.

    Args:
        model_cfg (dict): The model config.
        data_cfg (dict): The data config.
        env_cfg (dict): The environment config.
        log_option (dict): The log option.
    """
    # run the train module
    if env_cfg['multiprocessing_distributed']:
        mp.spawn(test_sub_module, nprocs=env_cfg['ngpus_per_node'], args=(
            model_cfg, data_cfg, env_cfg, log_option))
    else:
        test_sub_module(None, model_cfg=model_cfg,
                        data_cfg=data_cfg, env_cfg=env_cfg, log_option=log_option)


def test_sub_module(gpu_id, model_cfg, data_cfg, env_cfg, log_option):
    """The operation for sub test module.

    Args:
        gpu_id (int|None): The gpu id. This mean local rank in distributed learning.
        model_cfg (dict): The model config.
        data_cfg (dict): The data config.
        env_cfg (dict): The environment config.
        log_option (dict): The log option.
    """
    if gpu_id is None:
        logger = get_logger(**log_option)
    else:
        log_dir = os.path.dirname(log_option['log_dir'])
        log_option.update({'log_dir': os.path.join(
            log_dir, 'test_{}.log'.format(gpu_id))})
        logger = get_logger(**log_option)

    # set the gpu paramters
    logger.info('Set the gpu parameters.')
    is_cuda = torch.cuda.is_available()
    device = set_device(gpu_id)

    # distribution option
    if is_cuda and env_cfg['distributed']:
        logger.info('Set the rank.')
        set_rank(env_cfg, gpu_id)

        logger.info('Initalize the distributed process group.')
        init_process_group(
            env_cfg['dist_url'], env_cfg['dist_backend'], env_cfg['world_size'], env_cfg['rank'])

    # build model
    logger.info('Build the model.')
    model = build_model(model_cfg['model'], logger)

    # set model
    logger.info('Set the model.')
    model = set_model(model, device,
                      distributed=env_cfg['distributed'])

    # build optimizer & scheduler
    logger.info('Build the optimizer and learning rate scheduler.')
    optimizer = build_optimizer(model.parameters(), model_cfg['optimizer'])
    scheduler = build_scheduler(optimizer, model_cfg['scheduler'])

    # load the dataset
    if data_cfg['checkpoint'] is None:
        logger.warning('The checkpoint path should be in the config.')
        return
    elif not os.path.isfile(data_cfg['checkpoint']):
        logger.warning('The checkpoint path have wrong path.')
        return
    else:
        logger.info('Load the checkpoint : {}'.format(data_cfg['checkpoint']))
        checkpoint = torch.load(data_cfg['checkpoint'], map_location=device)

        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])

    # generate the dataset
    if data_cfg['dummy']:
        logger.info('Generate the dummy test dataset.')

        test_dataset = datasets.FakeData(
            100, (3, 227, 227), 1000, transforms.ToTensor())
    else:
        logger.info('Generate the test dataset.')

        test_pipeline = build_pipeline(data_cfg['pipeline'], mode="test")

        test_dataset = build_dataset(
            dataset=data_cfg['dataset'], root=data_cfg['test_dir'], transform=test_pipeline, split="test" if data_cfg['dataset'] in ["ImageNet"] else "validation")

    # generate the sampler
    test_sampler = None

    if env_cfg['distributed']:
        logger.info('Set the test_sampler sampler.')
        test_sampler = DistributedSampler(
            test_dataset, shuffle=False, drop_last=True)

    # load the dataset loader
    logger.info('Set the test data loader.')
    test_loader = DataLoader(dataset=test_dataset, batch_size=data_cfg['batch_size'],
                             shuffle=False, num_workers=env_cfg['workers'], pin_memory=True, sampler=test_sampler)

    # get test params
    test_params = build_param(model_cfg['params'], mode='test')

    logger.info('Test start.')
    test(test_loader, model, test_params, device,
         env_cfg['distributed'], env_cfg['world_size'])
    logger.info('Test done.')


def test(data_loader, model, params, device, distributed, world_size):
    """The operation for test.

    Args:
        data_loader (DataLoader): The torch dataloader.
        model (nn.Module): The torch model.
        params (dict): The paramter about test evaluation metrics for model.
        device (torch.device): The torch device.
        distributed (bool): The distribution option.
        world_size (int): The world size.
    """
    def run_test(loader):
        with torch.no_grad():
            end = time()

            for images, targets in loader:
                # data load time update
                data_time.update(time()-end)

                images = images.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)

                # get validation params and output[losses, evaluations, .., etc.]
                output = model(images, targets, return_loss=True, **params)

                for k in output:
                    if output[k].ndim:
                        output.update({k: torch.mean(output[k])})

                # output update
                metrics.update(output)

                # elapsed time update
                batch_time.update(time()-end)
                end = time()

    mode = 'test'
    data_time = AverageMeter('load_time', prefix=mode)
    batch_time = AverageMeter('infer_time', prefix=mode)
    metrics = MetricMeter(prefix=mode)

    model.eval()

    run_test(data_loader)

    if distributed:
        data_time.all_reduce(device=device)
        batch_time.all_reduce(device=device)
        metrics.all_reduce(device=device)

        # aux test set processing
        if len(data_loader.sampler)*world_size < len(data_loader.dataset):
            aux_test_dataset = Subset(data_loader.dataset, range(
                len(data_loader.sampler)*world_size, len(data_loader.dataset)))
            aux_test_loader = DataLoader(aux_test_dataset, batch_size=data_loader.batch_size/world_size,
                                         shuffle=False, num_workers=data_loader.workers, pin_memory=True)

            run_test(aux_test_loader)

    # display
    display_test(len(data_loader.dataset), metrics, data_time, batch_time)
