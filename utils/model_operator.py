import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import os

def model_para_num(model:nn.Module):
    res = 0
    for name,para in model.named_parameters():
        if para.grad is not None:
            res += para.data.nelement()
    return res
def model_para_small(model:nn.Module,apl = 1e-5):
    res = 0
    for name, para in model.named_parameters():
        if para.grad is not None:
            res += len(torch.where(para.data.view(-1)<apl)[0])
    return res



def set_model_grad_random(model:nn.Module):
    for name,para in model.named_parameters():
        if para.grad is not None:
            para.grad = torch.randn_like(para.grad)

def set_model_grad_to_weight(model:nn.Module,amount=0.95):
    for name,para in model.named_parameters():
        if para.grad is not None :
            if 'bn' in name:
                #print('cal bn')
                continue
            # this model has learnable parameter
            # find smallest para
            elenum = para.data.nelement()

            topk = torch.topk(torch.abs(para.data).view(-1), k=int(elenum*0.95), largest=False)
            #print(topk)
            para.grad.view(-1)[topk.indices] = para.data.view(-1)[topk.indices]


def calculate_grad_product(model1:nn.Module,model2:nn.Module):
    res = model1.conv1.weight.data.new_zeros([1])
    for (name1,para1),(name2,para2) in zip(model1.named_parameters(),model2.named_parameters()):
        assert name1 == name2,'using different model will cause error'
        if 'bn' in name1:
            continue
        if para1.grad is not None and para2.grad is not None:
            #print(para1.grad.device, para2.grad.device)
            res += torch.sum(para1.grad*para2.grad)

    #print(res)
    return res

def normalize_grad(model:nn.Module):
    norm = calculate_grad_product(model,model)
    norm = torch.sqrt(norm)
    for name,para in model.named_parameters():
        if para.grad is not None and 'bn' not in name:
            para.grad /= norm

def orthogonal(model1:nn.Module,model2:nn.Module):
    normalize_grad(model2)
    res = calculate_grad_product(model1,model2)
    for (name1,para1),(name2,para2) in zip(model1.named_parameters(),model2.named_parameters()):
        assert name1 == name2,'using different model will cause error'
        if 'bn' in name1:
            continue
        if para1.grad is not None and para2.grad is not None:
            para1.grad -= para2.grad*res

def distance(model1:nn.Module,model2:nn.Module):
    res = model1.conv1.weight.data.new_zeros([1])
    for (name1,para1),(name2,para2) in zip(model1.named_parameters(),model2.named_parameters()):
        assert name1 == name2,'using different model will cause error'
        res += torch.norm(para1.data - para2.data)**2
    res = torch.sqrt(res)
    return res



