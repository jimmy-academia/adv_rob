'''
adversarial training result vs image size: mnist and cifar10
'''

from data import get_dataset, get_dataloader
from adv import adversarial_training, pgd_attack
from config import empty_arg
from train import train_classifier, test_attack
from utils import dumpj

import time
import torch
import torchvision

from pathlib import Path
from tqdm import tqdm


def main(args):
	for dataset in ['mnist', 'cifar10']:
		print(f'=== working on dataset {dataset} ===')
		input_channel = 1 if dataset == 'mnist' else 3
		num_classes = 10
		
		result_path = Path('ckpt')/'testimgsize.json'
		trainset, testset = get_dataset(dataset, 32)
		trainloader, testloader = get_dataloader(trainset, testset)
		resnet = torchvision.models.resnet18(weights=None)
		resnet.conv1 = torch.nn.Conv2d(input_channel, 64, kernel_size=7, stride=2, padding=3, bias=False)
		resnet.fc = torch.nn.Linear(resnet.fc.in_features, num_classes)

		for it in range(1):
			eps = 0.3 if dataset == 'mnist' else 8/255
			eps = eps/4
			args.eps = eps
			print(args.eps)
			# train_acc = train_classifier(args, resnet, trainloader)
			train_acc = adversarial_training(args, resnet, trainloader)
			correct, adv_correct, total = test_attack(args, resnet, testloader, pgd_attack)
			test_acc, adv_acc = correct/total, adv_correct/total
			print(f'train_acc {train_acc}, test {test_acc}, attack {adv_acc}')

		eps = 0.3 if dataset == 'mnist' else 8/255
		args.eps = eps
		print(args.eps)
			
		for size in [8, 16, 32, 64, 128, 256, 512, 1024]:
			trainset, testset = get_dataset(dataset, size)
			trainloader, testloader = get_dataloader(trainset, testset)

			correct, adv_correct, total = test_attack(args, resnet, testloader, pgd_attack)
			print(f'size {size}, test {test_acc}, attack {adv_acc}')

def train(args):

	result_path = Path('ckpt')/'imgsize.json'

	Results = {}
	for dataset in ['mnist', 'cifar10']:
		Results[dataset] = {}
		for size in [8, 16, 32, 64, 128, 256, 512, 1024]:
			
			trainset, testset = get_dataset(dataset, size)

			trainloader, testloader = get_dataloader(trainset, testset)
			
			# pytorch model resnet
			resnet = torchvision.models.resnet18(weights=None)
			# adjust first layer to handle input image size
			input_channel = 1 if dataset == 'mnist' else 3
			eps = 0.3 if dataset == 'mnist' else 8/255
			args.eps = eps

			if size >= 32:
				resnet.conv1 = torch.nn.Conv2d(input_channel, 64, kernel_size=7, stride=2, padding=3, bias=False)
			else:
				resnet.conv1 = torch.nn.Conv2d(input_channel, 64, kernel_size=3, stride=1, padding=1, bias=False)
			num_classes = 10
			resnet.fc = torch.nn.Linear(resnet.fc.in_features, num_classes)

			# for images, labels in trainloader:
			# 	print(resnet(images).shape)
			# 	print(f'{dataset} with size {size} good to go')
			# 	break
			results = {'train_acc': [], 'test_acc': [], 'adv_acc': [], 'train_time': []}
			pbar = tqdm(range(20), desc=f'{dataset} with size {size}', ncols=88)
			train_time = 0
			for it in pbar:
				start = time.time()
				train_acc = adversarial_training(args, resnet, trainloader)
				train_time += time.time() - start
				correct, adv_correct, total = test_attack(args, resnet, testloader, pgd_attack)
				test_acc, adv_acc = correct/total, adv_correct/total
				results['train_acc'].append(train_acc)
				results['test_acc'].append(test_acc)
				results['adv_acc'].append(adv_acc)
				results['train_time'].append(train_time)
				pbar.set_postfix({'train': train_acc, 'test': test_acc, 'adv': adv_acc})

			Results[dataset][size] = results
			dumpj(Results, result_path)

if __name__ == '__main__':
	args = empty_arg()
	args.device = 0
	args.attack_iters = 100
	args.advtrain_lr = 1e-3
	main(args)
