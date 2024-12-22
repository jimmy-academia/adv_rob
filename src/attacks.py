import torch
import torch.nn as nn
from functools import partial
from autoattack import AutoAttack
from tqdm import tqdm
import logging

'''
autoattack ref: https://github.com/fra31/auto-attack
'''

def conduct_attack(args, model, test_loader, multi=False, do_test=True):
    if args.attack_type == 'fgsm':
        adv_perturb = fgsm_attack
    elif args.attack_type == 'pgd':
        adv_perturb = partial(pgd_attack, multi=multi)
    elif args.attack_type == 'pgd20':
        adv_perturb = partial(pgd_attack, multi=False, attack_iters=20)
    elif args.attack_type == 'aa_list':
        adv_perturb = partial(auto_attack, multi=True)
    elif args.attack_type == 'aa':
        adv_perturb = partial(auto_attack, multi=False)
    elif args.attack_type == 'square':
        adv_perturb = square_attack
    else:
        raise NotImplementedError(f'{args.attack_type} is not defined')

    total = test_correct = adv_correct = 0
    results = [0] * (3 if args.attack_type == 'pgd' else 4) if multi and args.attack_type in ['pgd', 'aa'] else None

    model.to(args.device)
    count = 0
    for images, labels in tqdm(test_loader, ncols=90, desc=f'conduct {args.attack_type} attack'):
        images, labels = images.to(args.device), labels.to(args.device)

        if do_test:
            pred = model(images)
            test_correct += float((pred.argmax(dim=1) == labels).sum())

        adv_images = adv_perturb(args, images, model, labels)
        if results is not None:
            for i, adv_image in enumerate(adv_images):
                adv_pred = model(adv_image)
                results[i]+= float((adv_pred.argmax(dim=1) == labels).sum())
        else:
            adv_pred = model(adv_images)
            adv_correct += float((adv_pred.argmax(dim=1) == labels).sum())
        total += len(labels)

        count+=1
        if args.attack_type == 'aa' and count >= 5:
            break

    logging.info(f'attack result: {[r/total for r in results] if results else adv_correct/total}')
    return test_correct, results if results else adv_correct, total

def fgsm_attack(args, primary, model, labels):
    variable = primary.clone().detach().requires_grad_(True)
    output = model(variable)
    loss = nn.CrossEntropyLoss()(output, labels)
    loss.backward()
    secondary = primary + args.eps * variable.grad.data.sign()
    return secondary.clamp(0., 1.)

def pgd_attack(args, primary, model, labels, sim=False, attack_iters=None, multi=False):
    secondary = primary + args.eps * (1- 2*torch.rand_like(primary)) # random perturbation within \pm args.eps
    secondary = secondary.detach()
    secondary = secondary.clamp(0., 1.)
    if attack_iters is None:
        attack_iters = 50 if multi else args.attack_iters

    results = []
    for step in range(1, attack_iters+1):
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

        if multi and step in {10, 20, 50}:
            results.append(secondary.clone())

    return results if multi else secondary

def auto_attack(args, primary, model, labels, _version='standard', multi=True):
    adversary = AutoAttack(model, norm='Linf', eps=args.eps, version=_version, verbose=False, device=args.device)
    if multi:
        adv_examples = adversary.run_standard_evaluation_individual(primary, labels, bs=primary.size(0))
        adv = [adv_examples[key] for key in ['apgd-ce', 'apgd-t', 'fab-t', 'square']]
    else:
        adv = adversary.run_standard_evaluation(primary, labels, bs=primary.size(0))
    return adv

def square_attack(args, primary, model, labels):
    adversary = AutoAttack(model, norm='Linf', eps=args.eps, version='custom', attacks_to_run=['square'], verbose=False, device=args.device)
    adv = adversary.run_standard_evaluation(primary, labels, bs=primary.size(0))
    return adv    
