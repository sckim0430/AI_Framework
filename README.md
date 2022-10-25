# Benchmark

## Purpose
- Improving pytorch-based program development skills
- The takeover process becomes more flexible.
- After writing only common code (train.py, test.py, etc.), it is possible to implement and apply only the desired model model. (reusability)
- When writing API code, only the model in the same format can be changed and used. (reusability)

## Configuration

### 1. Config

config files are largely composed of Model / Data / Environment related configuration based on json format.

<details>
<summary>Model Config</summary>

```
{
    "model": {
        "type": "AlexNet",
        "backbone": {
            "type": "AlexNet_Backbone",
            "in_channel": 3,
            "lrn_param": [
                5,
                1e-4,
                0.75,
                2.0
            ],
            "pretrained": null,
            "init_weight": true
        },
        "cls_head": {
            "type": "AlexNet_Head",
            "in_size": 6,
            "in_channel": 256,
            "num_class": 1000,
            "dropout_ratio": 0.5,
            "loss_cls": {
                "type": "CrossEntropyLoss",
                "loss_weight": 1.0
            },
            "avg_pooling": true,
            "multi_label": false,
            "init_weight": true
        }
    },
    "params": {
        "evaluation": {
            "train": {
                "accuracy_score": null,
                "top_k_accuracy_score": {
                    "k": 5
                },
                "precision_score": null,
                "recall_score": null
            },
            "validation": {
                "accuracy_score": null,
                "top_k_accuracy_score": {
                    "k": 5
                },
                "precision_score": null,
                "recall_score": null
            },
            "test": {
                "top_k_accuracy_score": {
                    "k": 5
                },
                "precision_score": null,
                "recall_score": null
            }
        },
        "loss": {
            "loss_cls": {
                "label_smoothing": 0.0,
                "weight": null
            }
        }
    },
    "optimizer": {
        "type": "SGD",
        "lr": 0.01,
        "momentum": 0.9,
        "weight_decay": 5e-4
    },
    "pipeline": {
        "train": {
            "RandomResizedCrop": {
                "size": 227
            },
            "RandomHorizontalFlip": null,
            "ToTensor": null,
            "Normalize": {
                "mean": [
                    0.485,
                    0.456,
                    0.406
                ],
                "std": [
                    0.229,
                    0.224,
                    0.225
                ]
            }
        },
        "validation": {
            "Resize": {
                "size": 256
            },
            "CenterCrop": {
                "size": 227
            },
            "ToTensor": null,
            "Normalize": {
                "mean": [
                    0.485,
                    0.456,
                    0.406
                ],
                "std": [
                    0.229,
                    0.224,
                    0.225
                ]
            }
        },
        "test": {
            "Resize": {
                "size": 256
            },
            "CenterCrop": {
                "size": 227
            },
            "ToTensor": null,
            "Normalize": {
                "mean": [
                    0.485,
                    0.456,
                    0.406
                ],
                "std": [
                    0.229,
                    0.224,
                    0.225
                ]
            }
        }
    }
}
```

</details>

<details>
<summary> Data Config </summary>

```
{
    "train_dir": "/ext/Dataset/ILSVRC2012_img_train/ILSVRC2012_img_train",
    "val_dir": "/ext/Dataset/ILSVRC2012_img_val/",
    "test_dir": "/ext/Dataset/ILSVRC2012_img_test/",
    "weight_dir": "/workspace/weight/",
    "weight_load": "/workspace/weight/load.pth",
    "log_dir": "/workspace/log/",
    "dummy": true,
    "batch_size": 64,
    "epochs": 100,
    "train_freq": 5,
    "val_freq": 5,
    "start_epoch": 0,
    "resume": null,
    "dataset": "ImageNet"
}

```
</details>

<details>

<summary> Environment Config </summary>

```
{
    "seed": null,
    "workers": 4,
    "multiprocessing_distributed": true,
    "distributed": true,
    "ngpus_per_node": 2,
    "world_size": 1,
    "rank": 0,
    "dist_url": "tcp://127.0.0.1:12345",
    "dist_backend": "nccl"
}


```

</details>

### 2. Build

It plays the role of calling the desired method or class from the config in json format.

The class or function specified in the config file can be called using the eval built-in function as follows.

<details>
<summary> Build </summary>

```
from torch.optim import *
from torchvision.transforms import *
from torchvision.datasets import *

from models.type import *
from models.module import *
from utils.parse import parse_type

def build(cfg, logger=None):
    """The operation for build.

    Args:
        cfg (dict): The input config.
        logger (logging.RootLogger): The logger. Defaults to None.

    Returns:
        nn.Module: The sub model object.
    """
    # parse type from config
    type, params = parse_type(cfg)

    if logger is not None:
        params.update({'logger': logger})

    return eval(type)(**params if params is not None else {}) 
```

</details>

### 3. Model


Model implementation is largely divided into Type / Module, and the Model Class is defined.

- Type: Define the entire Model Class
- Module: Sub Model Class that composes the Model defined in Type

For example, in general, Classification Task is composed of Backbone / Classification Head. Here, Backbone and Classification Head are implemented in Module, and Loss Class is additionally implemented. Finally, by calling these sub-classes in Type, the entire model class is implemented.

Common functions were implemented to inherit from Base Class, and individual functions were specified to be implemented as abstract methods.

### 4. Util

- Check : Check config file, data format, etc.
- Environment : Environment setting (distributed processing, gpu, torch.device, etc.)
- Log: Log settings (level, file path, format, etc.)
- Parse : Separate a specific key from the config
- Convert : Change data shape
- Checkpoint: Load and save checkpoints
- AverageMeter: Store and manage data values (loss, evaluation result, etc.)
- Display: data value output

### 5. Tool

It is a script that implements functions necessary for learning and inference.

### 6. Main

After loading and inspecting Config, Log setting, Environment setting, etc., the learning and inference module is called.

## Running image

run the train.py
![image](https://user-images.githubusercontent.com/63839581/197483003-129018ff-bb8e-444f-bc12-d2b4e78a03a8.png)

gpu check(nvidia-smi)
![image](https://user-images.githubusercontent.com/63839581/197483188-be3e5d56-85ee-42f0-8420-72e1337c8570.png)
