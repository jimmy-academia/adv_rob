import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from tqdm import tqdm

from data import get_dataloader
from customcnn import SimpleCNN
from eval import *
from utils import *

class NCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(NCNN, self).__init__()
        # Define the architecture
        self.conv1 = nn.Conv2d(1, 16, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc = nn.Linear(32 * 7 * 7, num_classes)  
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

def main():
    device = torch.device("cuda:"+str(0))
    

    Results = {'cluster':[[], []], 'normal':[[], []], 'transfer':[]}
    num_epochs = 10
    for epoch in range(num_epochs):
        modelpath = Path(f'ckpt/model.pth.{epoch}')
        infopath = Path(f'ckpt/info.pth.{epoch}')

        model = SimpleCNN()
        model.load_state_dict(torch.load(modelpath))
        Nmod = NCNN()

        Nmod.conv1.weight.data = model.conv1.weight.clone()
        Nmod.conv2.weight.data = model.conv2.weight.clone()
        Nmod.fc.load_state_dict(model.fc.state_dict())

        info = torch.load(infopath)
        model.conv1.patches, model.conv1.cluster_centers, __ = info[0]
        model.conv2.patches, model.conv2.cluster_centers, __ = info[1]
        model.temp = info[0][-1]
        model.set_temp()
        model.is_eval()

        Nmod.to(device)
        model.to(device)
        __, testloader = get_dataloader()    

        print()
        print(f'>>>>>>> Epoch {epoch} <<<<<<')
        print('[cluster model]')
        testacc, attackacc = exp_test(model, testloader, device)

        Results['cluster'][0].append(testacc)
        Results['cluster'][1].append(attackacc)

        print('[normal model]')
        testacc, attackacc = exp_test(Nmod, testloader, device)

        Results['normal'][0].append(testacc)
        Results['normal'][1].append(attackacc)

        print('[transfer from normal to cluster model]')
        transfer_acc = exp_transfer(model, Nmod, testloader, device)
        Results['transfer'].append(transfer_acc)

        dumpj(Results, 'result.json')




if __name__ == '__main__':
    main()