import sys 
sys.path.append('.')
import time
import torch 
import torch.nn as nn
from tqdm import tqdm
from pathlib import Path
from collections import defaultdict


from ipt.attacks import pgd_attack, square_attack
from ipt.networks import APTNet, SmallClassifier, Dummy, AptSizeClassifier
from ipt.train import test_attack
from ipt.data import get_dataset, get_dataloader

from config import default_arguments
from utils import dumpj, loadj

def do_adversarial_similarity_training(args, model, train_loader, test_loader):
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
        Record = run_tests(args, model, test_loader, Record)
        
    return Record

def do_adversarial_training(args, model, train_loader, test_loader):
    Record = defaultdict(list)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    elapsed_time = 0
    for epoch in range(args.num_epochs):
        start = time.time()
        pbar = tqdm(train_loader, ncols=88, desc='adversarial training')
        for images, labels in pbar:
            images, labels = images.to(args.device), labels.to(args.device)
            adv_images = pgd_attack(args, images, model, labels)
            output = model(adv_images)
            loss = nn.CrossEntropyLoss()(output, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            pbar.set_postfix({'acc': (output.argmax(dim=1) == labels).sum()/len(labels)})
        
        elapsed_time += time.time() - start
        Record['epoch'].append(epoch)
        Record['elapsed_time'].append(elapsed_time)
        Record = run_tests(args, model, test_loader, Record)
    return Record

def run_tests(args, model, test_loader, Record):
    correct, adv_correct, total = test_attack(args, model, test_loader, pgd_attack)
    print(f'[AST pgd] test acc: {correct/total:.4f}, adv acc: {adv_correct/total:.4f}...')
    Record['pgd_test_acc'].append(correct/total)
    Record['pgd_adv_acc'].append(adv_correct/total)

    correct, adv_correct, total = test_attack(args, model, test_loader, square_attack)
    print(f'[AST square] test acc: {correct/total:.4f}, adv acc: {adv_correct/total:.4f}...')
    Record['square_test_acc'].append(correct/total)
    Record['square_adv_acc'].append(adv_correct/total)
    return Record


def plot_records(record_paths, labels, x_axis='epoch'):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 2, figsize=(12, 8), dpi=150)
    for record_path, label in zip(record_paths, labels):
        record = loadj(record_path)
        ax[0, 0].plot(record[x_axis], record['pgd_test_acc'], label=label)
        ax[0, 1].plot(record[x_axis], record['pgd_adv_acc'], label=label)
        ax[1, 0].plot(record[x_axis], record['square_test_acc'], label=label)
        ax[1, 1].plot(record[x_axis], record['square_adv_acc'], label=label)
    
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
    args.ckpt_dir = Path('ckpt/1_compare_ast_at')
    args.ckpt_dir.mkdir(parents=True, exist_ok=True)

    train_set, test_set = get_dataset(args.dataset)
    train_loader, test_loader = get_dataloader(train_set, test_set, args.batch_size)
    
    aptnet_ast_record_path = args.ckpt_dir / 'aptent_ast.json'
    aptnet_at_record_path = args.ckpt_dir / 'aptent_at.json'
    cnn_at_record_path = args.ckpt_dir / 'cnn_at.json'

    record_paths = [aptnet_ast_record_path, aptnet_at_record_path, cnn_at_record_path]
    exist_list = [record_path for record_path in record_paths if record_path.exists()]

    ans = None
    if exist_list:
        ans = input(f"Record files {exist_list} already exist. Type `no` to skip. Press Enter to overwrite them.")

    if not exist_list or ans != 'no':        
        # conduct ast for aptnet
        aptnet = APTNet(args)
        classifier = SmallClassifier(args)
        model = Dummy(aptnet, classifier).to(args.device)

        aptnet_ast_record = do_adversarial_similarity_training(args, model, train_loader, test_loader)
        dumpj(aptnet_ast_record, aptnet_ast_record_path)
        # conduct at for aptnet
        aptnet_at_record = do_adversarial_training(args, model, train_loader, test_loader)
        dumpj(aptnet_at_record, aptnet_at_record_path)
        # conduct at for aptent-size CNN
        model = AptSizeClassifier(args).to(args.device)
        cnn_at_record = do_adversarial_training(args, model, train_loader, test_loader)
        dumpj(cnn_at_record, cnn_at_record_path)

    # creat the plots
    aptnet = APTNet(args)
    classifier = SmallClassifier(args)
    model = Dummy(aptnet, classifier)
    print(f"Number of trainable parameters in apt model: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    model = AptSizeClassifier(args)
    print(f"Number of trainable parameters in apt-size CNN model: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    plot_records(record_paths, ['APTNet AST', 'APTNet AT', 'CNN AT'])
    plot_records(record_paths, ['APTNet AST', 'APTNet AT', 'CNN AT'], 'elapsed_time')
    print('Done!')

if __name__ == '__main__':
    main()