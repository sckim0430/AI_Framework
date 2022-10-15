"""The AvergeMeter Implementation.
"""
from collections import defaultdict

import torch
import torch.distributed as dist

class AverageMeter(object):
    """The average meter of single value.

    Args:
        object (class): The base class.
    """
    def __init__(self,name=None,prefix='train'):
        """The initalization.

        Args:
            name (str): The value name. Defaults to None
            prefix (str, optional): The prefix. Defaults to 'train'
        """
        self.name = name
        self.prefix=prefix
        self.reset()

    def reset(self):
        """The operation for reset.
        """
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        """The operation for update of single value.

        Args:
            val (flaot|int|torch.tensor): The value.
            n (int, optional): The number of value. Defaults to 1.
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def all_reduce(self, device):
        """The operation for all reduce.

        Args:
            device (torch.device): The torch device option.
        """
        if device.type in ('cpu','mps'):
            return

        total = torch.tensor([self.sum, self.count], dtype=torch.float32, device=device)
        dist.all_reduce(total,dist.ReduceOp.SUM, async_op=False)
        self.sum, self.count = total.tolist()
        self.avg = self.sum / self.count

    def __str__(self):
        """The built-in function.

        Returns:
            str: The output string.
        """
        
        avg = self.avg
        if isinstance(self.avg,torch.tensor):
            avg =avg.item()

        if self.name is None:
            str_out = '{:.5f}'
            str_out.format(avg)
        else
            str_out = '{}_{} {:.5f}'
            str_out.format(self.prefix, self.name, avg)
        
        return str_out


class MetricMeter(object):
    """The average meter of dict.

    Args:
        object (class): The base class.
    """
    def __init__(self, prefix='train', delimiter='\t'):
        """The initalization.

        Args:
            prefix  (str, optional): The prefix. Defaults to 'train'
            delimiter (str, optional): The delimiter of dict. Defaults to '\t'.
        """
        self.meters = defaultdict(lambda:AverageMeter(name=None,prefix=prefix))
        self.delimiter = delimiter

    def update(self, input_dict):
        """The operation for update.

        Args:
            input_dict (dict): The input dict.

        Raises:
            TypeError: When the input dict's type is not dict occur the TypeError.
        """
        if input_dict is None:
            return

        if not isinstance(input_dict, dict):
            raise TypeError(
                'Input to MetricMeter.update() must be a dictionary'
            )

        for k, v in input_dict.items():
            if isinstance(v, torch.tensor):
                v = v.item()
            self.meters[k].update(v)

    def all_reduce(self,device):
        """The operation for all reduce.

        Args:
            device (torch.device): The torch device.
        """
        for name, meter in self.meters.items():
            meter.all_reduce(device=device)

    def __str__(self):
        """The built-in function.

        Returns:
            str: The output string.
        """
        output_str = []
        for name, meter in self.meters.items():
            output_str.append(
                '{}_{} {:.5f}'.format(meter.prefix, name, meter.avg)
            )
        return self.delimiter.join(output_str)