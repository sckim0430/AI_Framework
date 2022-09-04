"""Resource Configuration
"""

#dataloader option
batch_size = 64

#random option
seed = None

#cpu option
workers = 4

#gpu options
multiprocessing_distributed = False
gpu_id = None
world_size = -1
rank = -1
dist_url = 'tcp://220.70.46.201:2100'
dist_backend = 'nccl'