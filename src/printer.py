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
    pass
    

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

'''


        if not printed and not self.args.direct:
            correct_ind = pred.argmax(dim=1) == labels
            incorrect_ind = adv_pred.argmax(dim=1) != labels
            batch_indices = torch.nonzero(incorrect_ind*correct_ind).squeeze()[:5]
            
            sample_imgs+= [img for img in images[batch_indices]]
            sample_adv_imgs+= [img for img in adv_images[batch_indices]]
            

            final_cond = batch_idx == len(self.test_loader) - 1 and len(sample_imgs) > 0


            if len(sample_imgs) >= 5 or final_cond:
                
                sample_imgs = torch.stack(sample_imgs)
                sample_adv_imgs = torch.stack(sample_adv_imgs)

                all_vis = model_copy.iptnet.visualize_embeddings().cpu()
                if len(all_vis[:5]) < 5:
                    padding = torch.zeros(5-len(all_vis), self.args.channels, self.args.image_size, self.args.image_size)
                    all_vis = torch.cat([all_vis, padding], dim=0)

                train_imgs, __ = next(iter(self.train_loader))
                sample_imgs = sample_imgs[:5] 
                sample_adv_imgs = sample_adv_imgs[:5]  

                from printer import display_images_in_grid
                tmpfilepath = f'ckpt/tmp/{self.args.train_env}/{self.epoch}.jpg'
                os.makedirs(f'ckpt/tmp/{self.args.train_env}', exist_ok=True)
                # Perform reconstruction
                train_imgs = train_imgs[:5].to(self.args.device)  #5, 3, 32, 32
                train_recon = model_copy.iptnet(train_imgs)
                test_recon = model_copy.iptnet(sample_imgs)
                diffimages = sample_adv_imgs - sample_imgs
                adv_recons = model_copy.iptnet(sample_adv_imgs)


                display_images_in_grid(tmpfilepath, [all_vis, train_imgs, train_recon, sample_imgs, test_recon, diffimages, sample_adv_imgs, adv_recons], None, 1)

                printed = True 

'''