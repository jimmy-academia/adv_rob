import sys 
sys.path.append('.')
import time
import torch 
import torch.nn as nn
from tqdm import tqdm
from pathlib import Path
from collections import defaultdict
from functools import partial

from ipt.attacks import pgd_attack, auto_attack_dict
from ipt.networks import SmallClassifier, Dummy
from ipt.data import get_dataset, get_dataloader

from config import default_arguments
from utils import dumpj, loadj, check

# https://pytorch.org/vision/0.8/_modules/torchvision/models/resnet.html

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class LargeAPTNet(nn.Module):
    def __init__(self, args, _layers = [16, 64]):
        # in_size, out_size, hidden_layers
        # in_size: args.channels
        # out_size: args.vocab_size
        super(LargeAPTNet, self).__init__()
        self.args = args
        self.patch_numel = args.channels * args.patch_size * args.patch_size

        model_layers = []
        input_channel = args.channels
        for channel_size in _layers+[args.vocab_size]:
            stride = 2 if len(model_layers) == 0 else 1
            downsample = nn.Sequential(
                conv1x1(input_channel, channel_size, stride),
                nn.BatchNorm2d(channel_size),
            )
            model_layers.append(BasicBlock(input_channel, channel_size, stride, downsample))
            input_channel = channel_size

        self.tokenizer = nn.Sequential(*model_layers)
        self.embedding = nn.Embedding(args.vocab_size, self.patch_numel)
        self.softmax = nn.Softmax(-1)
        self.relu = nn.ReLU()

        self.inv_patcher = nn.ConvTranspose2d(args.patch_numel, args.channels, args.patch_size, args.patch_size)
        for param in self.inv_patcher.parameters():
            param.requires_grad = False

    def _initialize_weights(self):
        # Set the weights for an identity operation
        k, c, w, __ = self.patcher.weight.shape  # e.g., (12, 3, 2, 2)
        identity_filter = torch.zeros_like(self.patcher.weight)
        for i in range(k):
            channel = i % c    ## 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2,
            row = i//c % w     ## 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1,
            col = (i//(c*w))   ## 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 
            identity_filter[i, channel, row, col] = 1

        # Set the manually calculated weights and zero biases
        # self.patcher.weight.data = identity_filter
        # self.patcher.bias.data.zero_()

        self.inv_patcher.weight.data = identity_filter
        self.inv_patcher.bias.data.zero_()
    
    def inverse(self, x):
        assert len(x.shape) in [2, 3], 'Input shape should be either [batch_size, num_patches, patch_numel] or [num_patches, patch_numel]'
        if len(x.shape) == 2:
            x = x.unsqueeze(0)
        batch_size = x.size(0)
        x = x.contiguous().view(batch_size, self.args.num_patches_width, self.args.num_patches_width, self.args.patch_numel)  
        x = x.permute(0, 3, 1, 2)
        x = self.inv_patcher(x)
        return x

    def forward(self, x):
        x = self.tokenizer(x)
        x = x.permute(0, 2,3,1)
        x = x.view(x.size(0), -1, x.size(-1))
        x = torch.matmul(x, self.embedding.weight)
        x = self.inverse(x)
        return x
    
def dict_adversarial_similarity_training(args, model, train_loader, test_loader, dict_auto_attack):
    # assume: model.iptnet vs model.classifier
    Record = defaultdict(list)

    optimizer = torch.optim.Adam(model.iptnet.parameters(), lr=1e-3)
    opt_class = torch.optim.Adam(model.classifier.parameters(), lr=1e-3)

    elapsed_time = 0
    for epoch in range(args.num_epochs):
        start = time.time()    
        pbar = tqdm(train_loader, ncols=90, desc='ast:pretrain')
        for images, labels in pbar:
            images = images.to(args.device)
            labels = labels.to(args.device)
            adv_images = pgd_attack(args, images, model.iptnet, images, True)
            output = model.iptnet(adv_images)
            mseloss = nn.MSELoss()(output, images)
            optimizer.zero_grad()
            mseloss.backward()
            optimizer.step()
            pbar.set_postfix(loss=mseloss.item())

        pbar = tqdm(train_loader, ncols=88, desc='adv/sim training')
        
        for images, labels in pbar:
            images, labels = images.to(args.device), labels.to(args.device)
            adv_images = pgd_attack(args, images, model, labels, False)
            output = model.iptnet(adv_images)

            mseloss = nn.MSELoss()(output, images)
            optimizer.zero_grad()
            mseloss.backward()
            optimizer.step()

            cl_output = model.classifier(output.detach())
            cl_loss = nn.CrossEntropyLoss()(cl_output, labels)
            opt_class.zero_grad()
            cl_loss.backward()
            opt_class.step()

            adv_acc = (cl_output.argmax(dim=1) == labels).sum() / len(labels)
            pbar.set_postfix({'adv_acc': float(adv_acc)})

        elapsed_time += time.time() - start
        Record['epoch'].append(epoch)
        Record['elapsed_time'].append(elapsed_time)
        Record = run_auto_dict_tests(args, model, test_loader, Record, dict_auto_attack)
        
    return Record

def run_auto_dict_tests(args, model, test_loader, Record, dict_auto_attack):

    info_dict, atk_names = test_attack(args, model, test_loader, dict_auto_attack)
    total = info_dict['total']
    correct = info_dict['correct']
    Record[f'test_acc'].append(correct/total)
    print(f'test acc: {correct/total:.4f}')
    for atk_name in atk_names:
        adv_correct = info_dict[atk_name]
        print(f'[{atk_name}] adv acc: {adv_correct/total:.4f} ')
        Record[f'{atk_name}_adv_acc'].append(adv_correct/total)

    return Record

def test_attack(args, model, test_loader, dict_auto_attack, fast=True):
    info_dict = defaultdict(lambda: 0)
    model.eval()
    pbar = tqdm(test_loader, ncols=90, desc='test_attack', unit='batch', leave=False)
    for images, labels in pbar:
        images = images.to(args.device)
        labels = labels.to(args.device)
        pred = model(images)
        info_dict['correct'] += float((pred.argmax(dim=1) == labels).sum())
        info_dict['total'] += len(labels)
        adv_images_dict = dict_auto_attack(args, images, model, labels)
        for atk_name, adv_images in adv_images_dict.items():
            adv_pred = model(adv_images)
            info_dict[atk_name] += float((adv_pred.argmax(dim=1) == labels).sum())

        if fast:
            break
    return info_dict, adv_images_dict.keys()

def plot_records(record_paths, labels, x_axis='epoch'):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 2, figsize=(12, 8), dpi=150)
    for record_path, label in zip(record_paths, labels):
        record = loadj(record_path)
        ax[0, 0].plot(record[x_axis], record['pgd_test_acc'], label=label)
        ax[0, 1].plot(record[x_axis], record['pgd_adv_acc'], label=label)
        ax[1, 0].plot(record[x_axis], record['auto_test_acc'], label=label)
        ax[1, 1].plot(record[x_axis], record['auto_adv_acc'], label=label)
    
    ax[0, 0].set_title('PGD Test Accuracy')
    ax[0, 1].set_title('PGD Adv Accuracy')
    ax[1, 0].set_title('Square Test Accuracy')
    ax[1, 1].set_title('Square Adv Accuracy')
    for i in range(2):
        for j in range(2):
            ax[i, j].legend()
            ax[i, j].grid()
    plt.savefig(f'compare_ast_at_{x_axis}.jpg')

def main():
    args = default_arguments('mnist')
    args.num_epochs = 20
    args.ckpt_dir = Path('ckpt/3_larger_aptnet')
    args.ckpt_dir.mkdir(parents=True, exist_ok=True)

    train_set, test_set = get_dataset(args.dataset)
    train_loader, test_loader = get_dataloader(train_set, test_set, args.batch_size)
    
    record_path = args.ckpt_dir / 'apt_ast.json'
    autoplus = partial(auto_attack_dict, _version='plus')

    aptnet = LargeAPTNet(args)
    classifier = SmallClassifier(args)
    model = Dummy(aptnet, classifier).to(args.device)
    print(f"Number of trainable parameters in apt model: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
        
    ast_record = dict_adversarial_similarity_training(args, model, train_loader, test_loader, autoplus)
    dumpj(ast_record, record_path)
    print('Done!')

if __name__ == '__main__':
    main()

    # model = BasicBlock(3, 16, 2)
    # x = torch.randn(1, 3, 32, 32)
    # print(model(x).shape)
