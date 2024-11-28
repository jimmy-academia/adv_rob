import sys
import importlib

from utils import setup_logging, readf
import matplotlib.pyplot as plt
# from datasets import rev_norm_transform
from debug import *

def main_console():
    setup_logging()

    if len(sys.argv) < 2:
        logging.error("Usage: python printer.py main => gets script/main_exp.py")
        sys.exit(1)

    module = importlib.import_module('scripts.'+f'{sys.argv[1]}_exp')
    module.print_experiments()

def plot():
    

def display_images_in_grid(imgpath, image_list, labels=None, verbose=0):

    # Determine rows and columns
    # assert it is list
    # assert isinstance(image_list, list)

    num_rows = len(image_list) 
    num_cols = len(image_list[0])
    if verbose > 1:
        print(f'preparing grid image with {num_cols} columns and {num_rows} rows')
    plt.figure(figsize=(num_cols * 3, num_rows * 3))  # Adjust figure size

    for row in range(num_rows):
        for col in range(num_cols):
            plt.subplot(num_rows, num_cols, row * num_cols + col + 1)
            img = image_list[row][col]
            img = img.clamp(0, 1)
            plt.imshow(img.permute(1, 2, 0).cpu().detach().numpy())  # Assuming image in (C, H, W) format
            plt.xticks([])  # Remove x-axis ticks
            plt.yticks([])  # Remove y-axis ticks

    if labels is not None:
        for col in range(num_cols):
            plt.subplot(num_rows, num_cols, (num_rows - 1) * num_cols + col + 1)
            plt.xlabel(labels[col], fontsize=12)

    plt.tight_layout()
    plt.savefig(imgpath)
    plt.close()

    if verbose > 0:
        print()
        print('saved image grid in ', imgpath)


if __name__ == '__main__':
    main_console()