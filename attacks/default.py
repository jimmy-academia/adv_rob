import torch
import torch.nn as nn
from autoattack import AutoAttack

def pgd_attack(args, primary, model, labels, sim=False, attack_iters=None):
    secondary = primary + args.eps * (1- 2*torch.rand_like(primary)) # random perturbation within \pm args.eps
    secondary = secondary.detach()
    secondary = secondary.clamp(0., 1.)
    if attack_iters is None:
        attack_iters = args.attack_iters # custom evaluation
    for __ in range(attack_iters):
        variable = secondary.clone().detach().requires_grad_(True)
        output = model(variable)
        if sim: ## similarity
            loss = nn.MSELoss()(output, labels)
        elif type(labels) is int:
            loss = - nn.CrossEntropyLoss()(output, torch.ones(primary.size(0), device=args.device).long() * labels)
        else:
            loss = nn.CrossEntropyLoss()(output, labels)
        
        loss.backward()  # loss: image - model -> (label) - lossfunc -> value
        grad = variable.grad.data
        variable = variable + args.eps * torch.sign(grad)
        variable = variable.clamp(primary - args.eps, primary + args.eps)
        variable = variable.clamp(0., 1.) # for primary
        secondary = variable.detach()
    return secondary

def auto_attack(args, primary, model, labels, _version='standard'):
    adversary = AutoAttack(model, norm='Linf', eps=args.eps, version=_version, verbose=False, device=args.device)
    return adversary.run_standard_evaluation(primary, labels, bs=primary.size(0))
