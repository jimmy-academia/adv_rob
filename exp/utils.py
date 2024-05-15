import re
import json
import code 
import inspect

import time
import logging
from pathlib import Path

import random
import torch
import os
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


################################################

# read from file
def readf(path):
    with open(path, 'r') as f:
        return f.read()

# dump/load json file in better format

class NamespaceEncoder(json.JSONEncoder):
  def default(self, obj):
    if isinstance(obj, argparse.Namespace):
      return obj.__dict__
    else:
      return super().default(obj)

def dumpj(dictionary, filepath):
    with open(filepath, "w") as f:
        # json.dump(dictionary, f, indent=4)
        obj = json.dumps(dictionary, indent=4, cls=NamespaceEncoder)
        obj = re.sub(r'("|\d+),\s+', r'\1, ', obj)
        obj = re.sub(r'\[\n\s*("|\d+)', r'[\1', obj)
        obj = re.sub(r'("|\d+)\n\s*\]', r'\1]', obj)
        f.write(obj)

def loadj(filepath):
    with open(filepath) as f:
        return json.load(f)

################################################

def setup_logging(log_path='ckpt/logs', code=0, log_level=logging.INFO):
    """
    Sets up the logging configuration.
    """
    Path(log_path).mkdir(parents=True, exist_ok=True)
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(filename=f'{log_path}/experiment_{code}.log',
                        level=log_level,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

def record_runtime(start_time=None):
    """
    Records the runtime of a process. If start_time is provided, it calculates the elapsed time.
    """
    if start_time:
        elapsed_time = time.time() - start_time
        logging.info(f"Elapsed Time: {elapsed_time:.2f} seconds")
        return elapsed_time
    return time.time()


def print_progress_bar(iteration, total, prefix='', suffix='', decimals=1, length=100, fill='█', print_end="\r"):
    """
    Call in a loop to create a terminal progress bar.
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        print_end   - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + '-' * (length - filled_length)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end=print_end)
    # Print New Line on Complete
    if iteration == total: 
        print()

# Additional utility functions can be added here as needed.

def print_log(message):
    print(message)
    logging.info(message)

    