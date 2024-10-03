import time
import torch
import torch.nn.functional as F
from tqdm import tqdm
from attacks.default import pgd_attack
from train_env.base_eval import Base_trainer, test_attack

class TestTimeTrainer(Base_trainer):
    '''
    implement Test-Time Training
    '''
    def train(self):
        self.model.train()
        self.model.to(self.args.device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
        
        self.runtime = 0
        for self.epoch in range(1, self.num_epochs+1):
            self.total = self.correct = rotcorrect = 0
            pbar = tqdm(self.train_loader, ncols=88, desc='TTT_training', leave=False)

            for images, labels, rotimages, rotlabels in pbar:
                start_time = time.time()
                images, labels, rotimages, rotlabels = [x.to(self.args.device) for x in [images, labels, rotimages, rotlabels]]

                output = self.model(images)
                self.loss = torch.nn.CrossEntropyLoss()(output, labels)
                
                rotoutput = self.model.sshead(rotimages)
                self.loss += torch.nn.CrossEntropyLoss()(rotoutput, rotlabels)

                optimizer.zero_grad()
                self.loss.backward()
                optimizer.step()
                
                self.runtime += time.time() - start_time
                self.total += len(output)
                rotcorrect += (output.argmax(dim=1) == labels).sum().item()
                self.correct += (rotoutput.argmax(dim=1) == rotlabels).sum().item()
                pbar.set_postfix({'acc': self.correct/self.total, 'loss': self.loss.cpu().item(), 'rotac': rotcorrect/self.total})
                
            self.append_training_record()    
                
            self.periodic_check()
            


