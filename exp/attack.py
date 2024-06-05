import torch
import torch.nn as nn

def adv_perturb(primary, model, label, eps, num_iters):
    secondary = primary.clone().detach()
    for __ in range(num_iters):
        variable = secondary.clone().detach().requires_grad_(True)
        output = model(variable)
        loss = nn.CrossEntropyLoss()(output, label)
        loss.backward()
        grad = variable.grad.data
        variable = variable + eps * torch.sign(grad)
        variable = variable.clamp(primary - eps, primary + eps)
        variable = variable.clamp(-1, 1) # for images
        secondary = variable.detach()

    return secondary

