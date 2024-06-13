import code 
import inspect

import os
import random
import torch
import numpy as np

def set_seeds(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def check():
    current_frame = inspect.currentframe()
    caller_frame = current_frame.f_back
    caller_locals = caller_frame.f_locals
    caller_globals = caller_frame.f_globals
    for key in caller_globals:
        if key not in globals():
            globals()[key] = caller_globals[key]

    frame_info = inspect.getframeinfo(caller_frame)
    caller_file = frame_info.filename
    caller_line = frame_info.lineno

    print('### check function called...')
    print(f"Called from {caller_file}")
    print(f"--------->> at line {caller_line}")

    code.interact(local=dict(globals(), **caller_locals))

# read from file
def readf(path):
    with open(path, 'r') as f:
        return f.read()

def writef(path, content):
    with open(path, 'w') as f:
        f.write(content)

def awritef(path, content):
    with open(path, 'a') as f:
        f.write(content)