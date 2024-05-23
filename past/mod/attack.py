import torch
import torch.nn as nn

from utils import check


def adv_perturb(primary, model, pred, eps, num_iters):
    secondary = primary.clone().detach()
    for __ in range(num_iters):
        variable = secondary.clone().detach().requires_grad_(True)
        output = model(variable)
        # torch.softmax(output, dim=1)
        loss = nn.CrossEntropyLoss()(output, pred)
        try:
            loss.backward()
        except:
            check()
        grad = variable.grad.data
        variable = variable + eps * torch.sign(grad)
        variable = variable.clamp(primary - eps, primary + eps)
        variable = variable.clamp(-1, 1) # for images
        secondary = variable.detach()

    return secondary