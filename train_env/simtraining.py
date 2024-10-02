import time
import torch
import torch.nn.functional as F
from tqdm import tqdm

from attacks.default import pgd_attack
from train_env.eval import test_attack

class AdversarialSimilarityTrainer:
    def __init__(self, args, model, train_loader, test_loader):
        self.args = args
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader

    def train(self):
        self.model.train()
        self.model.to(self.args.device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
        
        training_records = {'epoch':[], 'accuracy':[], 'loss':[], 'runtime':[]}
        runtime = 0
        for epoch in range(1, 101):
            pbar = tqdm(self.train_loader, ncols=88, desc='adversarial training', leave=False)
            correct = total = 0
            for images, labels in pbar:
                start_time = time.time()
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                adv_images = pgd_attack(self.args, images, self.model, labels, attack_iters=7)
                # Forward pass
                output = self.model(adv_images)
                loss = torch.nn.CrossEntropyLoss()(output, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                runtime += time.time() - start_time
                total += len(output)
                correct += (output.argmax(dim=1) == labels).sum().item()
                pbar.set_postfix({'acc': correct/total, 'loss': loss.cpu().item()})
                
            training_records['epoch'].append(epoch)
            training_records['train_acc'].append(correct/total)
            
            test_correct, adv_correct, test_total = test_attack(args, model, self.test_loader, pgd_attack)
            training_records['test_acc'].append(test_correct/test_total)
            training_records['adv_acc'].append(adv_correct/test_total)
            training_records['loss'].append(loss.cpu().item())
            training_records['runtime'].append(runtime)

            print(f'Epoch [{epoch}/{self.num_epochs}], Loss: {loss.item():.4f}')
        return training_records



'''
def do_adversarial_similarity_training(args, model, train_loader, test_loader, adv_attacks, atk_names):
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
        Record = run_tests(args, model, test_loader, Record, adv_attacks, atk_names)
        
    return Record

'''