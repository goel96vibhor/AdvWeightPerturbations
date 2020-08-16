'''Some helper functions for PyTorch, including:
    - get_mean_and_std: calculate the mean and std value of dataset.
    - msr_init: net parameter initialization.
    - progress_bar: progress bar mimic xlua.progress.
'''
import torch
import os
import sys
import time
import math
from pathlib import Path
import numpy as np
import torch.nn as nn
import torch.nn.init as init
from models import *
# import matplotlib.pyplot as pl

if torch.cuda.is_available():
    USE_CUDA = True
else:
    USE_CUDA = False

def get_grad_norm_stats(model, epoch, iteration):
    
    print("asda asfaf")

def get_gradient_stats(model, epoch, iteration):
    layer_param = {}
    dir = Path("GradientStats") / Path(str(epoch)) / Path(str(iteration))
    if not os.path.exists(dir.as_posix()):
        os.makedirs(dir.as_posix())

    log_file = Path("GradientStats") / Path("gradientStats.log")
        # log the mean, variance, max and min
    f = open(log_file.as_posix(), "a")


    for name, param in model.named_parameters():
        if USE_CUDA:
            layer_param[name] = param.grad.view(-1).cpu().numpy() 
            # fig = pl.hist(param.grad.view(-1).cpu().numpy())
        else:
            layer_param[name] = param.grad.view(-1).numpy() 
            # fig = pl.hist(param.grad.view(-1).numpy())

        # pl.title('Histogram')
        # pl.xlabel("Value")
        # pl.ylabel("Frequency")
        # pl.savefig((dir / Path(name.replace(".",  "_") + ".png")).as_posix())
        # pl.clf()
        f.write(str(epoch) + "," + str(iteration) + "," + name + ',' + str(np.mean(layer_param[name])) + ',' + str(np.var(layer_param[name])) + ',' + str(np.max(layer_param[name])) + ',' + str(np.min(layer_param[name])) +"\n")
    f.close()
    return layer_param

def get_param_stats(list_of_params, param_name, buckets=50, take_abs=False):
    """
        list_of_params: expects a list of parameters,
        param_name: names of parameters
        buckets: number of buckets to split the data into 
    """ 
    bins = list(range(0, 100, 10))
    bin_counts = [0]*len(bins)
    final_data = torch.tensor([])
    if USE_CUDA:
        final_data = final_data.cuda()
    param_stats = {}
    for i,data in enumerate(list_of_params):
        if take_abs==False:
            final_data = torch.cat([final_data, data.grad.view(-1)])    
            param_stats[param_name[i]] = [np.percentile(data.grad.cpu().numpy(), bins[i]) for i in range(len(bins))]
        else:
            final_data = torch.cat([final_data, torch.abs(data.grad.view(-1))])
            param_stats[param_name[i]] = [np.percentile(torch.abs(data.grad).cpu().numpy(), bins[i]) for i in range(len(bins))]

    final_stats = [np.percentile(final_data.cpu().numpy(), bins[i]) for i in range(len(bins))]
    return param_stats, final_stats

def l2_norm(data):
    """
        l2 norm 
    """
    return ((data**2).sum().item())**(0.5)

def l1_norm(data):
    """ 
        l1 norm 
    """
    return torch.abs(data).sum().item()

def get_grad_norm(list_of_params, param_name):
    param_stats = {}
    for i, data in enumerate(list_of_params):
        l2 = l2_norm(data.grad)
        l1 = l1_norm(data.grad)
        param_stats[param_name[i]] = [l2,l1]
    return param_stats 

def log_stats(param_stats, bin_counts, grad_norm, epoch, iteration, dir, param_file="PerParamStats.log", bin_counts_file="OverallStats.log", grad_norm_file="GradNormStats.log"):
    with open(dir+"/"+param_file, "a") as writer:
        for param, val in param_stats.items():
            writer.write(str(epoch) + "," + str(iteration) + "," + param + "," + ",".join([str(x) for x in val])+"\n")
    
    with open(dir+"/"+bin_counts_file, "a") as writer:
        writer.write(str(epoch) + "," + str(iteration) + "," + ",".join([str(x) for x in bin_counts])+"\n")

    with open(dir+"/"+grad_norm_file, "a") as writer:
        for param, val in grad_norm.items():
            writer.write(str(epoch) + "," + str(iteration) + "," + param + "," + str(val[0]) + "," + str(val[1]) +"\n")

def get_mean_and_std(dataset):
    '''Compute the mean and std value of dataset.'''
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    print('==> Computing mean and std..')
    for inputs, targets in dataloader:
        for i in range(3):
            mean[i] += inputs[:,i,:,:].mean()
            std[i] += inputs[:,i,:,:].std()
    mean.div_(len(dataset))
    std.div_(len(dataset))
    return mean, std

def init_params(net):
    '''Init layer parameters.'''
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            init.kaiming_normal(m.weight, mode='fan_out')
            if m.bias:
                init.constant(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            init.constant(m.weight, 1)
            init.constant(m.bias, 0)
        elif isinstance(m, nn.Linear):
            init.normal(m.weight, std=1e-3)
            if m.bias:
                init.constant(m.bias, 0)


#_, term_width = os.popen('stty size', 'r').read().split()
#term_width = int(term_width)
term_width = 80
TOTAL_BAR_LENGTH = 50.
last_time = time.time()
begin_time = last_time
def progress_bar(current, total, msg=None):
    global last_time, begin_time
    if current == 0:
        begin_time = time.time()  # Reset for new bar.

    cur_len = int(TOTAL_BAR_LENGTH*current/total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

    sys.stdout.write(' [')
    for i in range(cur_len):
        sys.stdout.write('=')
    sys.stdout.write('>')
    for i in range(rest_len):
        sys.stdout.write('.')
    sys.stdout.write(']')

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time

    L = []
    L.append('  Step: %s' % format_time(step_time))
    L.append(' | Tot: %s' % format_time(tot_time))
    if msg:
        L.append(' | ' + msg)

    msg = ''.join(L)
    sys.stdout.write(msg)
    for i in range(term_width-int(TOTAL_BAR_LENGTH)-len(msg)-3):
        sys.stdout.write(' ')

    # Go back to the center of the bar.
    for i in range(term_width-int(TOTAL_BAR_LENGTH/2)+2):
        sys.stdout.write('\b')
    sys.stdout.write(' %d/%d ' % (current+1, total))

    if current < total-1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\n')
    sys.stdout.flush()

def format_time(seconds):
    days = int(seconds / 3600/24)
    seconds = seconds - days*3600*24
    hours = int(seconds / 3600)
    seconds = seconds - hours*3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes*60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds*1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f


def get_model(model_name, resnet_version):

      if resnet_version == 'own':

            if model_name == 'Resnet34':
                  net = ResNet34()
                  print("using model Resnet34")
            elif model_name == 'Resnet50':
                  net = ResNet50()
                  print("using model Resnet50")
            elif model_name == 'Resnet101':
                  net = ResNet101()
                  print("using model Resnet101")
            elif model_name == 'Resnet152':
                  net = ResNet152()
                  print("using model Resnet152")
            else:
                  net = ResNet18()
                  print("using model Resnet18")
            return net
      elif resnet_version == 'pretrained':

            if model_name == 'resnet20':
                  net = resnet20()
                  print("using pretrained model resnet20")
            elif model_name == 'resnet32':
                  net = resnet32()
                  print("using pretrained model resnet32")
            elif model_name == 'resnet44':
                  net = resnet44()
                  print("using pretrained model resnet44")
            elif model_name == 'resnet56':
                  net = resnet56()
                  print("using pretrained model resnet56")
            else:
                  net = resnet110()
                  print("using pretrained model resnet110")
            return net
      
      elif resnet_version == 'std_pretrained':
            kwargs = {}
            pretrained = False
            progress = False
            if model_name == 'resnet18':
                  net = resnet18(pretrained = pretrained, progress = progress, **kwargs)
                  print("using std pretrained model resnet18")
            elif model_name == 'resnet34':
                  net = resnet34(pretrained = pretrained, progress = progress, **kwargs)
                  print("using std pretrained model resnet34")
            elif model_name == 'resnet50':
                  net = resnet50(pretrained = pretrained, progress = progress, **kwargs)
                  print("using std pretrained model resnet50")
            return net

      elif resnet_version == 'cifar100':
            kwargs = {'num_classes': 100}
            pretrained = 'cifar100'
            if model_name == 'cifar_resnet20':
                  net = cifar_resnet20(pretrained, **kwargs)
                  print("using pretrained model cifar_resnet20")
            elif model_name == 'cifar_resnet32':
                  net = cifar_resnet32(pretrained, **kwargs)
                  print("using pretrained model cifar_resnet32")
            elif model_name == 'cifar_resnet44':
                  net = cifar_resnet44(pretrained, **kwargs)
                  print("using pretrained model cifar_resnet44")
            else:
                  net = cifar_resnet56(pretrained, **kwargs)
                  print("using pretrained model cifar_resnet56")
            return net
      elif resnet_version == 'mnist':
            kwargs = {'num_classes': 10}
            pretrained = 'mnist'
            if model_name == 'mnist_resnet18':
                  net = mnist_resnet18()
                  print("using model mnist_resnet18")
            elif model_name == 'mnist_resnet34':
                  net = mnist_resnet34()
                  print("using model mnist_resnet34")
            elif model_name == 'mnist_resnet50':
                  net = mnist_resnet50()
                  print("using model mnist_resnet50")
            else:
                  net = mnist_resnet101()
                  print("using model mnist_resnet101")
            return net
      elif resnet_version == 'preact':

            if model_name == 'PreActResnet34':
                  model = PreActResNet34().cuda()
                  print("using model Resnet34")
            elif model_name == 'PreActResnet50':
                  model = PreActResNet50().cuda()
                  print("using model Resnet50")
            elif model_name == 'PreActResnet101':
                  model = PreActResNet101().cuda()
                  print("using model Resnet101")
            elif model_name == 'PreActResnet152':
                  model = PreActResNet152().cuda()
                  print("using model Resnet152")
            elif model_name == 'PreActResnet18':
                  model = PreActResNet18().cuda()
                  print("using model Resnet18")
            return model
