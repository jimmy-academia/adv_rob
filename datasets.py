import numpy as np
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import Dataset, DataLoader

from utils import * # Rootdir, common_corruptions defined in utils

transform_Normalization_dict = {
    'svhn': transforms.Normalize((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970)),
    'mnist': transforms.Normalize((0.1307, 0.1307, 0.1307), (0.3081, 0.3081, 0.3081)),  # Expanded to 3 channels
    'cifar10': transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    'cifar100': transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    'imagenet': transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
}

def get_dataloader(args):
    data_transform = transforms.Compose([
        transforms.Resize(args.image_size),
        transforms.Grayscale(num_output_channels=3) if args.dataset == 'mnist' else (lambda x: x),
        transforms.ToTensor(),
        transform_Normalization_dict[args.dataset], 
    ])

    train_set = getattr(datasets, args.dataset.upper())(root=Rootdir, train=True, download=True, transform=data_transform)
    test_set = getattr(datasets, args.dataset.upper())(root=Rootdir, train=False, download=True, transform=data_transform)

    if args.test_time == 'none':
        print(f'>>>> Dataset: Clean {args.dataset}, no test time augmentation')
        pass
    elif args.test_time == 'standard':
        if args.test_domain == 'corrupt':
            print(f'>>>> Dataset: {args.dataset} + {args.test_domain}')
            if args.train_env == 'TTT':
                # don't augment train set for TTAdv
                train_set = RotatedDataset(train_set)
            test_set.data = prepare_corrupt_test_data(args)
    else:
        print('TODO: other test time settings')
        raise NotImplementedError

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)

    return train_loader, test_loader


def rotate_by_label(img, label):
    # Assumes that img is (nchannels, height, width)
    if label == 1:
        img = img.flip(2).transpose(1, 2) # rotate 90
    elif label == 2:
        img = img.flip(2).flip(1) # rotate 180
    elif label == 3:
        img = img.transpose(1, 2).flip(2) # rotate 270
    return img

def rotate_batch(batch_data, flag):
    length = len(batch_data)
    if flag == 'random':
        labels = torch.randint(4, (length,), dtype=torch.long)
    elif flag == 'expand':
        labels = torch.cat([torch.zeros(length, dtype=torch.long),
                    torch.zeros(length, dtype=torch.long) + 1,
                    torch.zeros(length, dtype=torch.long) + 2,
                    torch.zeros(length, dtype=torch.long) + 3])
        batch = batch.repeat((4,1,1,1))
    else: # fixed on one
        assert isinstance(label, int)
        labels = torch.zeros((length,), dtype=torch.long) + label

    rotated_batch_data = torch.stack([rotate_by_label(img, label) for img, label in zip(batch_data, labels)])
    return rotated_batch_data, labels

class RotatedDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img, label = self.dataset[index]
        target_ssh = np.random.randint(0, 4)  
        img_ssh = rotate_by_label(img, target_ssh)
        return img, label, img_ssh, target_ssh

def prepare_corrupt_test_data(args):
    dataset_mapping = {
        "cifar10": "CIFAR-10-C",
        "cifar100": "CIFAR-100-C"
    }
    perlvl_num = 10000
    if args.corrupt_type == 'all':
        all_data = []
        for corruption in common_corruptions:
            data = np.load(Rootdir/dataset_mapping[args.dataset]/f'{corruption}.npy')
            data = data[(args.corrupt_level-1)*perlvl_num: args.corrupt_level*perlvl_num]
            all_data.append(data)
        data = np.concatenate(all_data, axis=0)
    else:
        data = np.load(Rootdir/dataset_mapping[args.dataset]/f'{args.corrupt_type}.npy')
        data = data[(args.corrupt_level-1)*perlvl_num: args.corrupt_level*perlvl_num]
    return data
