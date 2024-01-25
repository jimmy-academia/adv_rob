import torch

for epoch in range(10):
    infopath = f'ckpt/info.pth.{epoch}'
    info = torch.load(infopath)
    print(info[0][-1])

print()
for epoch in range(10):
    modelpath = f'../../src/ckpt/t6c7/model.pth.{epoch}'
    data = torch.load(modelpath)
    print(data['info'][0][-1])