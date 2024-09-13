import torch 
import random

def random_sign(x):
    return torch.sign(2*torch.rand_like(x) - 1)

def calc_margin(model, x, y):
    logits = model(x)
    u = torch.arange(x.shape[0])
    y_corr = logits[u, y].clone()    
    logits[u, y] = -float('inf')
    y_others = logits.max(-1)[0]
    return y_corr - y_others

def make_shape(x, _len=2):
    return x if len(x.shape) == _len else x.unsqueeze(0)

def flat_square_attack(args, x, model, labels):
    with torch.no_grad():
        batch_size, patch_numel = x.shape
        x_best = torch.clamp(x + args.eps * random_sign(x), 0., 1.)
        margin_min = calc_margin(model, x_best, labels)
        
        eps = args.eps
        p = int(0.8 * args.patch_numel)
        N = 5000
        for _iter in range(N):
            idx_to_fool = (margin_min>0.0).nonzero().squeeze()
            if idx_to_fool.numel() == 0:
                return x_best
            
            x_curr = make_shape(x[idx_to_fool])
            x_best_curr = make_shape(x_best[idx_to_fool])
            y_curr = labels[idx_to_fool]
            margin_min_curr = margin_min[idx_to_fool]

            delta = torch.zeros(patch_numel).to(x.device)
            delta[torch.tensor(random.sample(range(patch_numel), p))] = eps * torch.sign(torch.ones(1)-2* random.random())

            x_new = x_best_curr + delta
            x_new = x_new.clamp(x_curr - args.eps, x_curr + args.eps)
            x_new = x_new.clamp(0., 1.)
            x_new = make_shape(x_new)

            margin = calc_margin(model, x_new, y_curr)
            idx_miscl = (margin <= 0.).float()
            idx_improved = (margin < margin_min_curr).float()
            idx_improved = torch.max(idx_improved, idx_miscl)

            # updates
            margin_min[idx_to_fool] = idx_improved * margin + (1. - idx_improved) * margin_min_curr
            idx_improved = idx_improved.reshape([-1, *[1]*len(x.shape[:-1])])
            x_best[idx_to_fool] = idx_improved * x_new + (1. - idx_improved) * x_best_curr
            
            ind_succ = (margin_min <= 0.).nonzero().squeeze()
            if ind_succ.numel() == batch_size:
                return x_best

            if (_iter +1) % (N//(p-1)) == 0:
                p -= 1
            if (_iter +1) % (N//int(args.eps*255 - 1)) == 0:
                eps -= 1/255
            
    return x_best