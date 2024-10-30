import torch
import torch.nn as nn
from autoattack import AutoAttack

def pgd_attack(args, primary, model, labels, sim=False):
    secondary = primary + args.eps * (1- 2*torch.rand_like(primary))
    secondary = secondary.detach()
    secondary = secondary.clamp(0., 1.)
    for __ in range(args.attack_iters):
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

def square_attack(args, primary, model, labels, random=False):
    if len(primary.shape) == 2:
        return flat_square_attack(args, primary, model, labels)
    else:
        version = 'rand' if random else 'standard'
        adversary = AutoAttack(model, norm='Linf', eps=args.eps, version='custom', attacks_to_run=['square'], verbose=False, device=args.device)
        return adversary.run_standard_evaluation(primary, labels, bs=primary.size(0))


def auto_attack(args, primary, model, labels, _version='standard'):
    adversary = AutoAttack(model, norm='Linf', eps=args.eps, version=_version, verbose=False, device=args.device)
    return adversary.run_standard_evaluation(primary, labels, bs=primary.size(0))


def auto_attack_dict(args, primary, model, labels, _version='standard'):
    adversary = AutoAttack(model, norm='Linf', eps=args.eps, version=_version, verbose=False, device=args.device)
    return adversary.run_standard_evaluation_individual(primary, labels, bs=primary.size(0))


def patch_square_attack(args, primary, model, labels):
    patch_square_attacker = PatchSquareAttack(model, norm='Linf', eps=args.eps, device=args.device)
    return patch_square_attacker.perturb(primary, labels)

