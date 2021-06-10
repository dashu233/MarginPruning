import torch
import torchvision.transforms as transforms
import torchvision
import torch.optim as optim
import torch.nn as nn
import torch.nn.utils.prune as prune
import torch.nn.functional as F
import argparse
import os
import path
from Model.resnet import resnet50
from pthflops import count_ops
import Model.resnet_cifar as resnet_cifar10
from utils.mask_schedule import RewarmSchedule,WarmSchedule,WarmScheduleLinear

# cifar10 model download from
# https://github.com/chenyaofo/pytorch-cifar-models/blob/master/pytorch_cifar_models/resnet.py

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='FetalHeart')
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--arch",type=str,default='resnet56')
    parser.add_argument("--pretrained",action='store_true')
    parser.add_argument('--train',action='store_true')
    parser.add_argument('--epoch',type=int,default=100)
    parser.add_argument('--finetune',action='store_true')
    parser.add_argument('--momentum',type=float,default=0.9)
    parser.add_argument('--weight_decay',type=float,default=1e-4)
    parser.add_argument('--prune_rate',type=float,default=0.9)
    parser.add_argument('--mask_rate',type=float,default=0.8)
    parser.add_argument('--mask_gamma',type=float,default=0.01)
    parser.add_argument('--use_new_method',action='store_true')
    parser.add_argument('--warm_up_mask',action='store_true')
    parser.add_argument('--warm_up_gamma', action='store_true')
    parser.add_argument('--start_epoch',type=int,default=5)
    parser.add_argument('--warm_iter',type=int,default=20)

    args = parser.parse_args()
    device = 'cuda'
    if args.arch == 'resnet50':

        if args.pretrained:
            model = resnet50(pretrained=True,device=device)
        else:
            model = resnet50(device=device)

    if args.arch == 'resnet20':
        model = resnet_cifar10.cifar10_resnet20()
        if args.pretrained:
            st = torch.load('Model/state_dicts/resnet20.pt')
            model.load_state_dict(st)

    if args.arch == 'resnet56':
        model = resnet_cifar10.cifar10_resnet56()
        if args.pretrained:
            st = torch.load('Model/state_dicts/resnet56.pt')
            model.load_state_dict(st)

    model.to(device)
    if not os.path.exists("result/"):
        os.mkdir("result/")
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.247, 0.2435, 0.2616))])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=256,
                                              shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=256,
                                             shuffle=False, num_workers=2)

    # for name,m in model.named_modules():
    #     print(name)
    #     if (isinstance(m,nn.Linear) or isinstance(m,nn.Conv2d)):
    #         print("small value in {}.weight: {:.2f}%({}/{})".format(name,
    #             100. * float(torch.sum(m.weight <= 1e-3))
    #             / float(m.weight.nelement()),torch.sum(m.weight <= 1e-3),m.weight.nelement()))



    if args.train:

        optimizer = optim.SGD(model.parameters(), momentum=args.momentum,
                              lr=args.lr, weight_decay=args.weight_decay)
        lr_schedule = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                           milestones=[40,80],
                                                           gamma=0.1)
        g_schedule = WarmScheduleLinear(args.start_epoch,
                                   args.start_epoch + args.warm_iter,
                                    args.mask_gamma,0.5)
        losser = nn.CrossEntropyLoss()

        @torch.no_grad()
        def set_model_grad(model:nn.Module,amount=0.95,gamma=0.1):
            for name,m in model.named_modules():
                if isinstance(m,nn.Linear) or isinstance(m,nn.Conv2d):
                    elenum = m.weight.data.nelement()
                    topk = torch.topk(torch.abs(m.weight.data).view(-1), k=int(elenum * amount), largest=False)
                    m.weight.grad.view(-1)[topk.indices] += gamma*m.weight.data.view(-1)[topk.indices]
        start_rate = 0
        start_gamma = args.mask_gamma
        mask_warm_up_end_ep = args.warm_iter + args.start_epoch
        start_gamma = start_gamma/float(2**mask_warm_up_end_ep)
        end_gamma = args.mask_gamma
        end_rate = args.mask_rate

        for e in range(args.epoch):
            train_loss = 0
            right_pred = 0
            all_pred = 0
            model.train()
            for i,(data,label) in enumerate(trainloader):
                data = data.to(device)
                label = label.to(device)
                output = model(data)
                optimizer.zero_grad()
                loss = losser(F.softmax(output),label)
                loss.backward()

                if args.use_new_method :
                    gm = g_schedule(e)
                    set_model_grad(model,end_rate,gm)
                optimizer.step()

                with torch.no_grad():
                    _,pred = torch.max(output,dim=1)
                    right_pred += len(torch.where(pred.view(-1)==label.view(-1))[0])
                    all_pred += len(label.view(-1))
                    train_loss += loss.item()

            print('epoch{} train_loss={:2f},train_acc={}/{}({:.2f}%)'.format(e,train_loss,
                                    right_pred,all_pred,100*right_pred/all_pred))
            print('gamma:',g_schedule(e))

            model.eval()
            with torch.no_grad():
                right_pred = 0
                all_pred = 0
                loss_all = 0
                for data, label in testloader:
                    data = data.to(device)
                    label = label.to(device)
                    output = model(data)
                    loss = losser(F.softmax(output), label)
                    loss_all  += loss.item()
                    _, pred = torch.max(output, dim=1)
                    right_pred += len(torch.where(pred.view(-1) == label.view(-1))[0])
                    all_pred += len(label.view(-1))

            print('epoch{} test_acc={}/{}({:.2f}%),loss:{}'.format(e,
                            right_pred, all_pred, 100*right_pred / all_pred,loss_all))
    # prune step
    model.to(device)
    model.eval()
    with torch.no_grad():
        right_pred = 0
        all_pred = 0
        for data, label in testloader:
            data = data.to(device)
            label = label.to(device)
            output = model(data)
            _,pred = torch.max(output, dim=1)
            right_pred += len(torch.where(pred.view(-1) == label.view(-1))[0])
            all_pred += len(label.view(-1))
        for data,label in testloader:
            data = data.to(device)
            #count_ops(model,data)
            break
    print('test_acc_before_prune={}/{}({:.2f}%)'
          .format(right_pred, all_pred,100* right_pred / all_pred))

    for name,m in model.named_modules():
        print(name)
        if (isinstance(m,nn.Linear) or isinstance(m,nn.Conv2d)):
            print("small value in {}.weight: {:.2f}%({}/{})".format(name,
                100. * float(torch.sum(m.weight <= 1e-3))
                / float(m.weight.nelement()),torch.sum(m.weight <= 1e-3),m.weight.nelement()))

    parameters_to_prune = []
    importance_score = {}
    #print(model.state_dict().keys())
    for name,m in model.named_modules():
        if (isinstance(m,nn.Linear) or isinstance(m,nn.Conv2d)):
            parameters_to_prune.append((m,'weight'))
            # bn_name = name[:-5] + 'bn' + name[-1]
            # print(bn_name)
            # if bn_name in model.state_dict():
            #     bn:nn.BatchNorm2d = model.state_dict()[bn_name]
            #     print(bn_name, bn.weight)
            #     importance_score[(m,'weight')] = getattr(m,'weight')*bn.weight
    #print('para_to_prune',parameters_to_prune)
    prune.global_unstructured(
        parameters_to_prune,
        prune.L1Unstructured,
        importance_score,
        amount=args.prune_rate
    )
    # for name,m in model.named_modules():
    #     print(name)
    #     if (isinstance(m,nn.Linear) or isinstance(m,nn.Conv2d)) and not name=='conv1':
    #         print("Sparsity in {}.weight: {:.2f}%({}/{})".format(name,
    #             100. * float(torch.sum(m.weight == 0))
    #             / float(m.weight.nelement()),torch.sum(m.weight == 0),m.weight.nelement()))
    # #for name,m in model.named_modules():
        #if isinstance(m,nn.Linear) or isinstance(m,nn.Conv2d):
            #parameters_to_prune.append((m,'weight'))
            #prune.remove(m,'weight')

    model.eval()
    with torch.no_grad():
        right_pred = 0
        all_pred = 0
        for data, label in testloader:
            data = data.to(device)
            label = label.to(device)
            output = model(data)
            _,pred= torch.max(output, dim=1)
            right_pred += len(torch.where(pred.view(-1) == label.view(-1))[0])
            all_pred += len(label.view(-1))
        for data,label in testloader:
            data = data.to(device)
            #count_ops(model,data)
            break

    print('test_acc_after_prune={}/{}({:.2f}%)'
          .format(right_pred, all_pred, 100*right_pred / all_pred))

    for e in range(10):
        train_loss = 0
        right_pred = 0
        all_pred = 0
        model.train()
        for i, (data, label) in enumerate(trainloader):
            data = data.to(device)
            label = label.to(device)
            output = model(data)
            optimizer.zero_grad()
            loss = losser(F.softmax(output), label)
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                _, pred = torch.max(output, dim=1)
                right_pred += len(torch.where(pred.view(-1) == label.view(-1))[0])
                all_pred += len(label.view(-1))
                train_loss += loss.item()

        print('epoch{} train_loss={:2f},train_acc={}/{}({:.2f}%)'.format(e, train_loss,
                                                                         right_pred, all_pred,
                                                                         100 * right_pred / all_pred))

        model.eval()
        with torch.no_grad():
            right_pred = 0
            all_pred = 0
            for data, label in testloader:
                data = data.to(device)
                label = label.to(device)
                output = model(data)
                _, pred = torch.max(output, dim=1)
                right_pred += len(torch.where(pred.view(-1) == label.view(-1))[0])
                all_pred += len(label.view(-1))

        print('epoch{} test_acc={}/{}({:.2f}%)'.format(e,
                                                       right_pred, all_pred, 100 * right_pred / all_pred))
