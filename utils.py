
'''
Torch settings and helper functions
'''
import os
import random
import torch
import numpy as np

import re
import json
import argparse

torch.set_printoptions(sci_mode=False)

def set_seeds(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def readf(path):
    with open(path, 'r') as f:
        return f.read()

## rootdir setting for storing dataset from pytorch
for dir_ in ['ckpt', 'cache']:
    os.makedirs(dir_, exist_ok=True)
default_rootdir_logpath = 'cache/rootdir'
warn_msg = f"Warning: {default_rootdir_logpath} does not exist. Setting to './cache'"
Rootdir = readf(default_rootdir_logpath) if os.path.exists(default_rootdir_logpath) else (print(warn_msg) or './cache')

def writef(path, content):
    with open(path, 'w') as f:
        f.write(content)

def awritef(path, content):
    with open(path, 'a') as f:
        f.write(content)

class NamespaceEncoder(json.JSONEncoder):
  def default(self, obj):
    if isinstance(obj, argparse.Namespace):
      return obj.__dict__
    else:
      return super().default(obj)

def dumpj(dictionary, filepath):
    with open(filepath, "w") as f:
        obj = json.dumps(dictionary, indent=4, cls=NamespaceEncoder)
        obj = re.sub(r'("|\d+),\s+', r'\1, ', obj)
        obj = re.sub(r'\[\n\s*("|\d+)', r'[\1', obj)
        obj = re.sub(r'("|\d+)\n\s*\]', r'\1]', obj)
        f.write(obj)

def loadj(filepath):
    with open(filepath) as f:
        return json.load(f)