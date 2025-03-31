import torch
import torch.nn as nn

# Targeted FGSM 공격
def fgsm_targeted(model, x, target, eps):
    x_adv = x.clone().detach().requires_grad_(True)
    loss = nn.CrossEntropyLoss()(model(x_adv), target)
    loss.backward()
    return torch.clamp(x_adv - eps*x_adv.grad.sign(), 0, 1)

# Untargeted FGSM 공격
def fgsm_untargeted(model, x, label, eps):
    x_adv = x.clone().detach().requires_grad_(True)
    loss = nn.CrossEntropyLoss()(model(x_adv), label)
    loss.backward()
    return torch.clamp(x_adv + eps*x_adv.grad.sign(), 0, 1)

# PGD 공격 (targeted)
def pgd_targeted(model, x, target, k, eps, eps_step):
    x_adv = x.clone().detach()
    for _ in range(k):
        x_adv.requires_grad_(True)
        loss = nn.CrossEntropyLoss()(model(x_adv), target)
        loss.backward()
        with torch.no_grad():
            x_adv -= eps_step*x_adv.grad.sign()
            x_adv = torch.max(torch.min(x_adv,x+eps),x-eps)
            x_adv = torch.clamp(x_adv,0,1)
    return x_adv

# PGD 공격 (untargeted)
def pgd_untargeted(model, x, label, k, eps, eps_step):
    x_adv = x.clone().detach()
    for _ in range(k):
        x_adv.requires_grad_(True)
        loss = nn.CrossEntropyLoss()(model(x_adv), label)
        loss.backward()
        with torch.no_grad():
            x_adv += eps_step*x_adv.grad.sign()
            x_adv = torch.max(torch.min(x_adv,x+eps),x-eps)
            x_adv = torch.clamp(x_adv,0,1)
    return x_adv
