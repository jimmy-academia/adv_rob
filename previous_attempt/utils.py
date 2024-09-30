import sys
import code 
import inspect
import traceback


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


original_excepthook = sys.excepthook


class FrameNavigator:
    def __init__(self, frames):
        self.frames = frames
        self.current_frame_index = 0
        self.update_context(self.current_frame_index)

    def update_context(self, index):
        self.current_frame_index = index
        frame = self.frames[index]
        self.locals = frame.f_locals.copy()
        self.globals = frame.f_globals
        frame_info = inspect.getframeinfo(frame)
        self.filename = frame_info.filename
        self.lineno = frame_info.lineno
        print(f"Switched to frame {index}: {self.filename} at line {self.lineno}")
        # Update interactive console locals
        self.update_interactive_locals()

    def update_interactive_locals(self):
        # Update the locals dictionary in the interactive console
        interactive_locals.clear()
        interactive_locals.update(self.locals)
        interactive_locals.update({'nv': self, 'ls': list_vars})

    def next(self):
        if self.current_frame_index < len(self.frames) - 1:
            self.update_context(self.current_frame_index + 1)
        else:
            print("Already at the newest frame")

    def prev(self):
        if self.current_frame_index > 0:
            self.update_context(self.current_frame_index - 1)
        else:
            print("Already at the oldest frame")

    def list(self):
        print("Frames:")
        for i, frame in enumerate(self.frames):
            frame_info = inspect.getframeinfo(frame)
            prefix = "* " if i == self.current_frame_index else ""
            print(f"{prefix}frame {i}: {frame_info.filename} at line {frame_info.lineno}")


def is_user_code(frame):
    # Check if the frame is from user code by comparing the file path
    filename = frame.f_globals["__file__"]
    return not filename.startswith(sys.prefix)

def list_vars():
    print("Local variables in the current frame:")
    for var, val in interactive_locals.items():
        if not var.startswith("__") and not callable(val):
            print(f"{var}: {val}")

def syscheck():
    # Restore the original excepthook
    sys.excepthook = original_excepthook

    # Get the last traceback
    tb = sys.last_traceback
    user_frames = []

    # Collect all user frames
    while tb:
        frame = tb.tb_frame
        if is_user_code(frame):
            user_frames.append(frame)
        tb = tb.tb_next

    if not user_frames:
        print("No user frames found")
        return

    global interactive_locals
    interactive_locals = {}

    navigator = FrameNavigator(user_frames)

    # Interactive console with frame navigation
    banner = (
        "\n"
        "=== Interactive mode ===\n"
        "Use 'nv.next()' to go to the next frame, "
        "Use 'nv.prev()' to go to the previous frame.\n"
        "Use 'nv.list()' to list all frames.\n"
        "Use 'ls()' to list local variables in the current frame.\n"
        "Local variables of the current frame are accessible.\n"
    )


    def interact():
        navigator.update_interactive_locals()
        code.interact(banner, local=interactive_locals)

    interact()

def custom_excepthook(exctype, value, tb):
    if exctype == KeyboardInterrupt:
        print("KeyboardInterrupt caught. Exiting cleanly.")
        sys.exit(0)
    else:
        traceback.print_exception(exctype, value, tb)
        sys.last_traceback = tb  # Save the last traceback to use in check()
        syscheck()


def debug_mode():
    sys.excepthook = custom_excepthook


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