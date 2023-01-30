#
# Copyright (c) 2023 Salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
#
import torch
import random
import os
import numpy as np


def tensor2flatten_arr(tensor):
    """convert tensor to flat numpy array

    Args:
        tensor (torch.Tensor): tensor object

    Returns:
        numpy array: flat numpy array
    """
    return tensor.data.cpu().numpy().reshape(-1)


def seed_everything(seed=1234):
    """fix the random seeds throughout the python environment

    Args:
        seed (int, optional): Seed value. Defaults to 1234.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def set_device(gpu: int = None):
    """set device (cpu or gpu). Use -1 to specify cpu.
    If not manually set device would be automatically set to gpu
    if gpu is available otherwise cpu would be used

    Args:
        gpu (int, optional): device number of gpu (use -1 for cpu).
         Defaults to None.

    Returns:
        torch.device : torch device type object
    """
    if not gpu:
        if torch.cuda.is_available():
            device = torch.device("cuda:" + str(torch.cuda.current_device()))
        else:
            device = torch.device("cpu")
    else:
        device = torch.device("cuda:" + str(gpu))
    return device
