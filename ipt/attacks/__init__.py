import torch
import torch.nn as nn

from autoattack import AutoAttack
from ipt.attacks.flat_square import flat_square_attack
from ipt.attacks.patch_square import PatchSquareAttack

def pgd_attack(args, primary, model, labels):
    secondary = primary.clone().detach()
    for __ in range(args.attack_iters):
        variable = secondary.clone().detach().requires_grad_(True)
        output = model(variable)
        loss = nn.CrossEntropyLoss()(output, labels)
        loss.backward()
        grad = variable.grad.data
        variable = variable + args.eps * torch.sign(grad)
        variable = variable.clamp(primary - args.eps, primary + args.eps)
        variable = variable.clamp(0., 1.) # for primary
        secondary = variable.detach()

    return secondary

def square_attack(args, primary, model, labels):
    if len(primary.shape) == 2:
        return flat_square_attack(args, primary, model, labels)
    else:
        adversary = AutoAttack(model, norm='Linf', eps=args.eps, version='custom', attacks_to_run=['square'], verbose=False, device=args.device)
        return adversary.run_standard_evaluation(primary, labels, bs=primary.size(0))

def patch_square_attack(args, primary, model, labels):
    patch_square_attacker = PatchSquareAttack(model, norm='Linf', eps=args.eps, device=args.device)
    return patch_square_attacker.perturb(primary, labels)