import torch
from .eval import exp_test
from tqdm import tqdm

from collections import defaultdict
from utils import *

def run_experiment(args, model, trainloader, testloader):

    #train
    log_data = defaultdict(list)
    model = model.create_equivalent_normal_cnn()
    model.to(args.device)

    runtime = 0
    for epoch in range(args.epochs):
        model.set_mode('train')
        running_loss = correct = total = 0
        
        pbar = tqdm(trainloader, ncols=88, desc='train', leave=False)

        start_time = record_runtime()
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

        for imgs, labels in pbar:
            imgs = imgs.to(args.device)
            labels = labels.to(args.device)

            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            accuracy = correct / total
            pbar.set_postfix(acc=accuracy)

        runtime += record_runtime(start_time)

        training_loss = running_loss/len(trainloader)

        log_data['epoch'].append(epoch)
        log_data['train_time'].append(runtime)
        log_data['train_loss'].append(training_loss)
        log_data['train_acc'].append(accuracy)
        print_log(f'Epoch {epoch+1}/{args.epochs}, Training Loss: {training_loss:.3f}, Accuracy: {accuracy * 100:.2f}%, runtime:{runtime:.3f}')

        model.save_param(args.checkpoint_dir, epoch)
        
        testacc, attackacc = exp_test(args, model, testloader)
        print_log(f'Test Accuracy: {testacc * 100:.2f}%, Adversarial Accuracy: {attackacc * 100:.2f}%')
        log_data['test_acc'].append(testacc)
        log_data['attack_acc'].append(attackacc)
        
    dumpj(log_data, args.checkpoint_dir/'log.json')

