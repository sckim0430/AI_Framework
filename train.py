"""Train Script For Classification Model
"""
import random
import warnings
import os

import argparse
from configparser import ConfigParser

import torch
import torch.backends.cudnn as cudnn
import torch.multiprocessing as mp
import torch.distributed as dist

import models

def parse_args():
    """Generate Arguments
    Returns:
        argparse.Namespace : arguments
    Notice:
        We Only Use ini Format File
    """

    parser = argparse.ArgumentParser(description='Pytorch Imagenet Train')
    parser.add_argument('--model_config_dir',default='/home/sckim/AI_Framework/configs/models/alexnet/alexnet.ini',help='model config path')
    parser.add_argument('--train_config_dir',default='/home/sckim/AI_Framework/configs/train_config.ini',help='train config path')
    parser.add_argument('--resource_config_dir',default='/home/sckim/AI_Framework/configs/resource_config.ini',help='resource config path')
    args = parser.parse_args()

    return args

def main():
    """Main Worker Controller
    """
    args = parse_args()
    model_config = ConfigParser()
    train_config = ConfigParser()
    resource_config = ConfigParser()
    model_config.read(args.model_config_dir)
    train_config.read(args.train_config_dir)
    resource_config.read(args.resource_config_dir)

    # if args.seed is not None:
    #     random.seed(args.seed)
    #     torch.manual_seed(args.seed)
    #     cudnn.deterministic = True
    #     warnings.warn('You have chosen to seed training. '
    #                   'This will turn on the CUDNN deterministic setting, '
    #                   'which can slow down your training considerably! '
    #                   'You may see unexpected behavior when restarting '
    #                   'from checkpoints.')
    
    # if args.gpu is not None:
    #     warnings.warn('You have chosen a specific GPU. This will completely '
    #                   'disable data parallelism.')

    
    # #여섯 가지의 경우의 수
    # #1. cpu
    # #2. single gpu, single process, single node
    # #3. single gpu, multi process, multi node
    # #4. multi gpu, single process, single node
    # #5. multi gpu, multi process, single node
    # #6. multi gpu, multi process, multi node
    
    # #dist_url : 프로세스 그룹을 초기화하는 방법을 명시한 url을 의미한다.
    # #0. (2,4)와 같은 single process에서는 distributed training을 하지 않기 떄문에, world_size와 rank를 무시한다.
    # #1. dist_url이 env://로 지정된 경우에는 환경 변수에 world_size, rank, master_port, master_ip가 지정된 것으로 간주한다.
    # #2. dist_url에 world_size, rank 정보가 포함되어있지 않은 경우에은 world_size, rank를 초기화 시켜야한다.
    # #3. dist_url에 world_size, rank 정보가 포함되어있는 경우에는 world_size, rank를 초기화 시키기 않아도 된다. => 이 코드에선 사용 x
    # #4. dist_url을 사용하지 않는 경우에는 store 옵션을 사용해야되며, world_size, rank는 필수로 초기화 시켜야한다. => 이 코드에선 사용 x
    # if args.dist_url == 'env://' and args.world_size == -1:
    #     args.world_size = int(os.environ["WORLD_SIZE"])

    # #multiprocessing_distributed 옵션은 이 노드에서 멀티 프로세스 동작시킬지에 대한 옵션(5,6)
    # #world_size>1인 경우는 멀티 노드인 경우를 의미한다.(3)
    # #즉, distributed는 멀티 프로세스 옵션
    # args.distributed = args.world_size>1 or args.multiprocessing_distributed

    # #노드 당 gpu 수를 의미
    # #즉, 현재 PC에 존재하는 gpu 수를 의미한다.
    # #gpu 1개 당 1개의 프로세스를 할당
    # ngpus_per_node = torch.cuda.device_count()
    
    # #world_size는 노드 수에서 프로세스의 수를 의미하게 되어, (5,6)의 경우에는
    # #world_size = ngpus_per_node * world_size로 재정의한다. (rank도 재정의)
    # if args.multiprocessing_distributed:
    #     args.world_size = ngpus_per_node * args.world_size
    #     mp.sqawn(main_worker,nprocs=ngpus_per_node,args=(ngpus_per_node,args))
    # else:
    #     main_worker(args.gpu,ngpus_per_node,args)

def main_worker(gpu,ngpus_per_node,args):
    #gpu : 현재 프로세스에서 사용하는 gpu id
    #이때, gpu는 None이거나 특정 gpu가 지정된 경우이다.
    #None인 경우에는 gpu를 사용할 수 있다면 전체 gpu 목록에 모델 weight를 할당한다. (model.cuda())
    #특정 gpu가 지정된 경우에는 해당 gpu에만 모델 weight를 할당한다. (torch.cuda.set_device(gpu_id), model.cuda(gpu_id))
    args.gpu = gpu

    if args.gpu is not None:
        print("Use GPU : {} for training".format(args.gpu))
    
    if args.distributed:
        #world_size와 마찬가지로 dist_url이 env://인 경우에는 환경 변수의 값을 참조하여 구한다.
        if args.dist_url=='env://' and args.rank==-1:
            args.rank = int(os.environ['RANK'])
        
        #args.rank는 전체 노드중에서 현재 노드의 우선순위를 의미하므로,
        #현재 노드의 우선순위 * 각 노드당 gpu 수 + 현재 gpu id(순위)로 갱신한다.
        #현재 노드의 우선순위에서 프로세스의 순위로 변경된다.
        if args.multiprocessing_distributed:
            args.rank = args.rank*ngpus_per_node+gpu

        dist.init_process_group(backend=args.dist_backend,init_method=args.dist_url,world_size=args.world_size,rank=args.rank)

    #create model
    if args.pretrained:
        # print("=> using pre-trained model : '{}'".format(os.path.basename(args.config_dir)[:-3]))
        pass
    
    print(models.__dict__)

if __name__=='__main__':
    main()