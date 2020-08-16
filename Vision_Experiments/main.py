'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms
import numpy as np
import os, random
import argparse
import time
from PIL import Image
# import matplotlib.pyplot as plt
from models import *
import cifar_own
import sys
from adv_extension import *
from utils import progress_bar, get_gradient_stats, get_param_stats, log_stats, get_grad_norm, get_model

# from scipy.misc import toimage
parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
parser.add_argument('--optim', default='adam', type=str, help='which optimizer to use', choices=['adam', 'sgd'])
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--adversary', '-adv', action='store_true', help='train for adversary')
parser.add_argument('--model_name', default='Resnet18', type=str, help='name for the model')
parser.add_argument('--checkpoint_dir', default='checkpoint', type=str, help='name for the checkpoint directory')
parser.add_argument('--resnet_version', default='own', type=str, help='name for the model',
                    choices=['own', 'pretrained', 'cifar100', 'mnist', 'std_pretrained', 'preact'])
parser.add_argument('--dataset', default='cifar10', type=str, help='which dataset to use',
                    choices=['cifar10', 'cifar100', 'mnist'])
parser.add_argument('--job_name', default='', type=str, help='name for the job')
parser.add_argument('--num_epochs', default=10, type=int, help='number of epochs to train')
parser.add_argument('--num_adv_batch', default=10, type=int, help='number of adv batches to use for training')
parser.add_argument('--weight_reg_param', default=0.0005, type=float,
                    help='value of regularization parameter to control total weight change')
parser.add_argument('--weight_adv_loss', default=1, type=float,
                    help='value of weight parameter to proportionate adversarial loss impact')
parser.add_argument('--model_eps', type=float, default=0.005, help='model parameters eps value for projection')
parser.add_argument('--train_model_eps_ball', action='store_true', help='project model or not in eps ball')
parser.add_argument('--lp_norm', default='inf', type=str, help='norm for projecting the model weights')
parser.add_argument('--eps_in_percent', default=0, type=int, help='model epsilon is in percent or not')
parser.add_argument('--seed', default=0, type=int, help='Random seed')
parser.add_argument('--num_pixels_changed', default=100, type=int,
                    help='number of pixels to change for backdoor generation')
parser.add_argument('--use_random_pattern', default=0, type=int, help='whether to use random pattenr for backdoor generation')
parser.add_argument('--set_model_labels', default=0, type=int, help='whether to use labels from original model for training')
parser.add_argument('--use_full_nonadv_data', default=0, type=int, help='whether to use full non adv data for backdoor training')
parser.add_argument('--random_pattern_mode', default=1, type=int,
                    help='what random pattern type to use: 0 to set to zero, 1 to random set, 2 for random add'
                    , choices=[0, 1, 2])
parser.add_argument('--pattern_eps', type=float, default=1, help='pattern eps value to specify max change in pixel value for backdoor')
parser.add_argument('--save_epoch_checkpoints', default=0, type=int, help = 'save model checkpoints in each epoch')
args = parser.parse_args()
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
random.seed(args.seed)
torch.backends.cudnn.deterministic = True

starttime = time.time()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
best_adv_acc = 0
start_epoch = 0  # start from epoch 0 or last checkpoint epoch
model_name = args.model_name
job_name = args.job_name
num_epochs = args.num_epochs
num_adv_batch = args.num_adv_batch
weight_reg_param = args.weight_reg_param
num_pixels_changed = args.num_pixels_changed
use_random_pattern = args.use_random_pattern
if not job_name:
    log_dir = "GradientStatsPercentile_Abs_Norm"
else:
    log_dir = "GradientStatsPercentile_Abs_Norm/" + job_name
if not os.path.isdir(log_dir):
    os.mkdir(log_dir)

# Data
print("Starting job: %s for model: %s, with log directory: %s and arguments ..." % (job_name, model_name, log_dir))
print(sys.argv[1:])
print('==> Preparing data..')
transform_train = transforms.Compose([
    #     transforms.RandomCrop(32, padding=4),
    #     transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
if args.dataset == 'cifar100':
    trainset = cifar_own.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
    testset = cifar_own.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
    trainset_length = trainset.__len__()
    testset_length = testset.__len__()
    image_dim = 32
    num_channels = 3
elif args.dataset == 'mnist':
    mnist = torchvision.datasets.MNIST(download=True, train=True, root="./data").train_data.float()
    print(mnist.mean()/255.0)
    print(mnist.std() / 255.0)
    data_transform = transforms.Compose([ transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize((mnist.mean()/255,), (mnist.std()/255,))])
    trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=data_transform)
    testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=data_transform)
    trainset_length = trainset.__len__()
    testset_length = testset.__len__()
    image_dim = 224
    num_channels = 1
else:
    trainset = cifar_own.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    testset = cifar_own.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    trainset_length = trainset.__len__()
    testset_length = testset.__len__()
    image_dim = 32
    num_channels = 3
# trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
# print(trainset.train_list)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=False, num_workers=2,
                                          worker_init_fn=np.random.seed(args.seed))

# testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2,
                                         worker_init_fn=np.random.seed(args.seed))

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# Model
print('==> Building model..')
# net = VGG('VGG19')

net = get_model(model_name, args.resnet_version)

# net = PreActResNet18()
# net = GoogLeNet()
# net = DenseNet121()
# net = ResNeXt29_2x64d()
# net = MobileNet()
# net = MobileNetV2()
# net = DPN92()
# net = ShuffleNetG2()
# net = SENet18()
# net = ShuffleNetV2(1)
# net = EfficientNetB0()

if args.resnet_version != 'std_pretrained':
      net = net.to(device)

      if device == 'cuda':
            net = torch.nn.DataParallel(net)
            cudnn.benchmark = True
      else:
            net = torch.nn.DataParallel(net)
if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir(args.checkpoint_dir), 'Error: no checkpoint directory found!'
    assert os.path.isdir(args.checkpoint_dir + '/' + model_name), 'Error: no checkpoint directory found!'
    checkpoint = torch.load(args.checkpoint_dir + '/' + model_name + '/ckpt.pth', map_location=torch.device(device))
    saved_model = get_model(model_name, args.resnet_version)
    
    if args.resnet_version != 'std_pretrained':
      saved_model = saved_model.to(device)
      if device == 'cuda':
            saved_model = torch.nn.DataParallel(saved_model)
            cudnn.benchmark = True
      else:
            saved_model = torch.nn.DataParallel(saved_model)
    if args.resnet_version == 'own':
        net.load_state_dict(checkpoint['net'])
        saved_model.load_state_dict(checkpoint['net'])
        best_acc = checkpoint['acc']
        start_epoch = checkpoint['epoch']
    elif args.resnet_version == 'pretrained':
        net.load_state_dict(checkpoint['state_dict'])
        saved_model.load_state_dict(checkpoint['state_dict'])
        best_acc = checkpoint['best_prec1']
    elif args.resnet_version == 'std_pretrained':
        net.load_state_dict(checkpoint)
        saved_model.load_state_dict(checkpoint)
        net = net.to(device)
        saved_model = saved_model.to(device)
        if device == 'cuda':
            net = torch.nn.DataParallel(net)
            saved_model = torch.nn.DataParallel(saved_model)
            cudnn.benchmark = True
        else:
            net = torch.nn.DataParallel(net)
            saved_model = torch.nn.DataParallel(saved_model)

      #   best_acc = checkpoint['best_prec1']
        print("loaded model from standard pretrained directory")
    elif args.resnet_version == 'mnist':
        net.load_state_dict(checkpoint['net'])
        saved_model.load_state_dict(checkpoint['net'])
        best_acc = checkpoint['acc']
        start_epoch = checkpoint['epoch']
    #     else:
    #       net = get_model(model_name, args.resnet_version)
    #       net = net.to(device)
    #       print(checkpoint.keys())
    #       net.load_state_dict(checkpoint)

    #       if device == 'cuda':
    #             net = torch.nn.DataParallel(net)
    #             cudnn.benchmark = True
    #       else:
    #             net = torch.nn.DataParallel(net)

    # saved_model.load_state_dict(checkpoint)
    # best_acc = checkpoint['best_acc1']
    # start_epoch = checkpoint['epoch']
    print("model loaded")

# print(net.parameters())
# for param in net.parameters():
#     print(param)
if args.adversary:
    adv_extender = Adv_extend(image_dim, num_pixels_changed=num_pixels_changed, use_random_pattern=args.use_random_pattern,
                              random_pattern_mode=args.random_pattern_mode, pattern_eps = args.pattern_eps, num_channels = num_channels)
    adv_trainset = adv_extender.extend_dataset(trainset, shuffle=False, train=True, num_batches_toadd=num_adv_batch,
                                               use_full_nonadv_data=args.use_full_nonadv_data, set_model_labels = args.set_model_labels, orig_model = saved_model)
    adv_trainset_length = len(adv_trainset)
    print("new train length:%d" % (len(adv_trainset)))
    adv_testset = adv_extender.extend_dataset(testset, shuffle=False, train=False, concat=False, use_full_nonadv_data=True, set_model_labels = False)
    adv_testset_length = len(adv_testset)
    print("new test length:%d" % (len(adv_testset)))

criterion = nn.CrossEntropyLoss()
if args.optim == 'adam':
      print("using adam optimizer ....")
      optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
else:
      print("using sgd optimizer ....")
      optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)


# saved_parameters = copy.deepcopy(net.parameters())

def _init_fn(worker_id):
    np.random.seed(args.seed)


def calculate_tensor_percentile(t: torch.tensor, q: float):
    """
    Return the ``q``-th percentile of the flattened input tensor's data.

    CAUTION:
     * Needs PyTorch >= 1.1.0, as ``torch.kthvalue()`` is used.
     * Values are not interpolated, which corresponds to
       ``numpy.percentile(..., interpolation="nearest")``.

    :param t: Input tensor.
    :param q: Percentile to compute, which must be between 0 and 100 inclusive.
    :return: Resulting value (scalar).
    """
    # Note that ``kthvalue()`` works one-based, i.e. the first sorted value
    # indeed corresponds to k=1, not k=0! Use float(q) instead of q directly,
    # so that ``round()`` returns an integer, even if q is a np.float32.
    k = 1 + round(.01 * float(q) * (t.numel() - 1))
#     print("k for percentile: %d" %(k))
    result = t.view(-1).kthvalue(k).values.item()
    return result


def calc_indiv_loss(y, targets):
    temp = F.softmax(y)
    loss = [-torch.log(temp[i][targets[i].item()]) for i in range(y.size(0))]
    #     loss = F.cross_entropy(y, targets, reduction = 'None')
    return torch.stack(loss)


def projection_lp_norm(cur_model, orig_model, model_eps, percentiles, lp_norm='inf', print_norm=False,
                       calc_diff_percentiles=True):
    diff_percents = dict.fromkeys(percentiles)
    norm_diff_percents = dict.fromkeys(percentiles)
    with torch.no_grad():
        l2_loss = nn.MSELoss(size_average=False, reduction='mean')
        param_diff = torch.empty(0, 1)
        param_diff = torch.flatten(param_diff)
        param_diff_norm = torch.empty(0, 1)
        param_diff_norm = torch.flatten(param_diff_norm)
        param_diff = param_diff.to(device)
        param_diff_norm = param_diff_norm.to(device)
        if lp_norm == 'inf':
            for i, (cur_param, orig_param) in enumerate(zip(cur_model.parameters(), orig_model.parameters())):
                #     cur_param.data = orig_param.data - torch.clamp(orig_param.data - cur_param.data, -1*model_eps, model_eps)
                if args.eps_in_percent:

                    # cur_param.data = torch.clamp(cur_param.data, orig_param.data(1.0 - eps/100.0), orig_param.data(1.0 + eps/100.0))
                    cur_param.data = torch.where(cur_param < orig_param * (1.0 - model_eps / 100.0),
                                                 orig_param * (1.0 - model_eps / 100.0), cur_param)
                    cur_param.data = torch.where(cur_param > orig_param * (1.0 + model_eps / 100.0),
                                                 orig_param * (1.0 + model_eps / 100.0), cur_param)
                else:
                    cur_param.data = orig_param.data - torch.clamp(orig_param.data - cur_param.data, -1 * model_eps,
                                                                   model_eps)
                    # cur_param.data = torch.clamp(cur_param.data, orig_param.data - eps, orig_param.data + eps)

                if calc_diff_percentiles:
                    layer_diff = torch.abs(torch.flatten(cur_param - orig_param))
                    layer_diff_norm = torch.div(layer_diff, torch.abs(torch.flatten(orig_param)))
                  #   if(i==1):
                  #         print(torch.abs(torch.flatten(cur_param)))
                  #         print(layer_diff)
                  #         print(torch.abs(torch.flatten(orig_param)))
                  #         print(layer_diff_norm)
                    param_diff = torch.cat([layer_diff, param_diff], dim=0)
                    param_diff_norm = torch.cat([layer_diff_norm, param_diff_norm], dim=0)
                  #   print(param_diff_norm.shape)

                if (print_norm and i < 5):
                    print(cur_param.shape)
                    print(l2_loss(cur_param, orig_param))
                    print(torch.norm(cur_param - orig_param, p=float("inf")))
                    print("")

        if model_eps == 0:
            for (n1, param1), (n2, param2) in zip(cur_model.named_parameters(), orig_model.named_parameters()):
                #     print(param1, param2)
                assert torch.all(param1.data == param2.data) == True
        if calc_diff_percentiles:
            for i in percentiles:
                diff_percents[i] = calculate_tensor_percentile(param_diff, i)
                norm_diff_percents[i] = calculate_tensor_percentile(param_diff_norm, i)
    return [diff_percents, norm_diff_percents]


# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    count = 0

    conv_param_names = []
    conv_params = []
    for name, param in net.named_parameters():
        if "conv" in name:
            conv_params += [param]
            conv_param_names += [name]

    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        # if random.uniform(0,1) < 0.2 and count < 5:
        #     count +=1
        #     get_gradient_stats(net, epoch, batch_idx)

        if batch_idx % 10 == 0:
            # conv params
            param_stats, bin_counts = get_param_stats(conv_params, conv_param_names)
            grad_norm_stats = get_grad_norm(conv_params, conv_param_names)
            log_stats(param_stats, bin_counts, grad_norm_stats, dir=log_dir, epoch=epoch, iteration=batch_idx)
            param_stats, bin_counts = get_param_stats(conv_params, conv_param_names, take_abs=True)
            grad_norm_stats = get_grad_norm(conv_params, conv_param_names)
            log_stats(param_stats, bin_counts, grad_norm_stats, dir=log_dir, epoch=epoch, iteration=batch_idx,
                      param_file="PerParamStatsAbs.log", bin_counts_file="OverallStatsAbs.log",
                      grad_norm_file="GradNormStatsAbs.log")

        optimizer.step()
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d) | Time: %.2f'
                     % (
                     train_loss / (batch_idx + 1), 100. * correct / total, correct, total, (time.time() - starttime)))


def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d) | Time: %.2f'
                         % (test_loss / (batch_idx + 1), 100. * correct / total, correct, total,
                            (time.time() - starttime)))

    # Save checkpoint.
    acc = 100. * correct / total
    
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint/' + model_name):
            os.mkdir('checkpoint/' + model_name)
        torch.save(state, './checkpoint/' + model_name + '/ckpt.pth')
        best_acc = acc
    if (args.save_epoch_checkpoints == 1):
            print("saving checkpoint at epoch %d .... " %(epoch))
            if not os.path.exists(args.checkpoint_dir):
                  os.mkdir(args.checkpoint_dir)
            model_dir = os.path.join(args.checkpoint_dir, args.model_name)
            if not os.path.exists(model_dir):
                  os.mkdir(model_dir)
            state = {
                  'net': net.state_dict(),
                  'acc': acc,
                  'epoch': epoch,
            }
            torch.save(state, os.path.join(model_dir, str('ckpt' + '_'+ str(epoch) + '.pth') ))


def adv_train(epoch, dataloader):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    count = 0
    percentiles = ['0', '25', '50', '75', '90']
    diff_percents = dict.fromkeys(percentiles)
    norm_diff_percents = dict.fromkeys(percentiles)
    conv_param_names = []
    conv_params = []
    for name, param in net.named_parameters():
        if "conv" in name:
            conv_params += [param]
            conv_param_names += [name]

    sum = 0
    max_batch = int((adv_trainset_length-1)/128)
    for batch_idx, (inputs, labels) in enumerate(dataloader):
        # print(batch_idx)
        sum += inputs.shape[0]
        
        # if batch_idx != 493:
        #     continue
        inputs, labels = inputs.to(device), labels.to(device)

        targets = torch.abs(labels).to(device)
        sample_weights = torch.ones(targets.shape, dtype=torch.float64, device=device)

        sample_weights = torch.where(labels >= 0, sample_weights, sample_weights * args.weight_adv_loss)

        # print(inputs[0].shape)

        sample_id = 14
        if batch_idx != 3 or batch_idx != 493:
            show_image = False
            # continue
        else:
            show_image = True

        l2_loss = nn.MSELoss(size_average=False, reduction='sum')
        reg_loss = 0
        num_paramsA = 0
        num_paramsB = 0
        for param_id, (paramA, paramB) in enumerate(zip(net.parameters(), saved_model.parameters())):
            reg_loss += l2_loss(paramA, paramB)
            # if (batch_idx == 0 and param_id < 5):
            #     print("reg loss:%0.8f, param_id:%d" % (reg_loss, param_id))
            #     print(torch.norm(paramA - paramB, p=float("inf")))
            num_paramsA += np.prod(list(paramA.shape))
            num_paramsB += np.prod(list(paramB.shape))
            # print(paramA.shape)
            # print(paramB.shape)

        factor = weight_reg_param
        # loss += factor * reg_loss
        # print(reg_loss)

        optimizer.zero_grad()
        outputs = net(inputs)
        #   loss = criterion(outputs, targets)
        indiv_loss = calc_indiv_loss(outputs, targets)
        indiv_loss = indiv_loss * sample_weights

        mean_loss = torch.mean(indiv_loss)
        #   print("mean loss %0.9f" %(loss))
        loss = criterion(outputs, targets)
        #   print("criterion loss %0.9f" %(loss))
        loss += reg_loss * factor
        loss.backward()
        # if random.uniform(0,1) < 0.2 and count < 5:
        #     count +=1
        #     get_gradient_stats(net, epoch, batch_idx)

        if batch_idx % 10 == 0:
            # conv params
            # print(sample_weights)
            # print(labels)
            # print("mean loss %0.9f" %(loss))
            # print("criterion loss %0.9f" %(loss))
            param_stats, bin_counts = get_param_stats(conv_params, conv_param_names)
            grad_norm_stats = get_grad_norm(conv_params, conv_param_names)

            log_stats(param_stats, bin_counts, grad_norm_stats, dir=log_dir, epoch=epoch,
                      iteration=batch_idx)
            param_stats, bin_counts = get_param_stats(conv_params, conv_param_names, take_abs=True)
            grad_norm_stats = get_grad_norm(conv_params, conv_param_names)
            log_stats(param_stats, bin_counts, grad_norm_stats, dir=log_dir, epoch=epoch,
                      iteration=batch_idx, param_file="PerParamStatsAbs.log", bin_counts_file="OverallStatsAbs.log",
                      grad_norm_file="GradNormStatsAbs.log")
            # print( 'Loss: %.3f | Acc: %.3f%% (%d/%d) | Time: %.2f' % (test_loss/(batch_idx+1), 100.*correct/total, correct, total, (time.time()-starttime)))
            # print(batch_idx)

        optimizer.step()

        if args.train_model_eps_ball:
            print_norm = False
            if (batch_idx == 0):
                print("Training model in epsilon ball")
                print_norm = True
            else:
                print_norm = False
            if (batch_idx == max_batch):
                print("Calculating diff percentiles")
                calc_diff_percentiles = True
            else:
                calc_diff_percentiles = False
            [diff_percents, norm_diff_percents] = projection_lp_norm(net, saved_model, args.model_eps, percentiles,
                                                                     args.lp_norm, print_norm, calc_diff_percentiles = calc_diff_percentiles)
        #     print("projected batch")

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    #   time.sleep(5)

    #   progress_bar(batch_idx, len(dataloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d) | Param_diff: %.3f | tot_par_a: %d | Time: %.2f'
    #                % (train_loss / (batch_idx + 1), 100. * correct / total, correct, total, reg_loss, num_paramsA, (time.time()-starttime)))

    diff_tensor = torch.cat(
        [(param_1 - param_2).view(-1) for param_1, param_2 in zip(net.parameters(), saved_model.parameters())], dim=0)
    x = torch.cat([param_2.view(-1) for param_2 in saved_model.parameters()], dim=0)

    l2_norm_diff = float(torch.norm(diff_tensor))
    l2_norm_orig = float(torch.norm(x))

    linf_norm_diff = float(torch.norm(diff_tensor, float('inf')))
    linf_norm_orig = float(torch.norm(x, float('inf')))

    l1_norm_diff = float(torch.norm(diff_tensor, 1))
    l1_norm_orig = float(torch.norm(x, 1))
    print("max batches: %d" %(max_batch))
    print('Loss: %.3f | Acc: %.3f%% (%d/%d) | Param_diff: %.3f | tot_par_a: %d | Time: %.2f' % (
    train_loss / (batch_idx + 1), 100. * correct / total, correct, total, reg_loss, num_paramsA,
    (time.time() - starttime)))

    result_dict = {'Loss': float(train_loss / (batch_idx + 1)), 'Acc': 100. * correct / total, 'Correct': correct,
                   'Total': total,
                   'Param_diff': reg_loss, 'Total_param': num_paramsA, 'Time': time.time() - starttime,
                   'l2_norm_diff': l2_norm_diff, 'l2_norm_orig': l2_norm_orig,
                   'linf_norm_diff': linf_norm_diff, 'linf_norm_orig': linf_norm_orig, 'l1_norm_diff': l1_norm_diff,
                   'l1_norm_orig': l1_norm_orig}
    print(diff_percents)
    print(norm_diff_percents)
#     print(l1_norm_diff)
#     print(l1_norm_orig)
#     print(result_dict)
    return diff_percents, norm_diff_percents, result_dict
    # break


#     print("batch_total %d" % (sum))

def adv_test(epoch, dataloader, save_chkpoint=False):
    global best_adv_acc
    net.eval()
    test_loss = 0
    orig_test_loss = 0
    correct = 0
    total = 0
    adv_sim = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            if (batch_idx % 10 == 0):
                print(batch_idx)
            inputs, targets = inputs.to(device), targets.to(device)
            targets = torch.abs(targets)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            orig_outputs = saved_model(inputs)
            orig_loss = criterion(orig_outputs, targets)
            orig_test_loss += orig_loss.item()
            _, orig_predicted = orig_outputs.max(1)
            adv_sim += predicted.eq(orig_predicted).sum().item()

            # progress_bar(batch_idx, len(dataloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d) | Time: %.2f'
            #     % (test_loss/(batch_idx+1), 100.*correct/total, correct, total, (time.time()-starttime)))
            # if(batch_idx%10==0):
        print('Loss: %.3f | Acc: %.3f%% (%d/%d) | Adv_sim: %.3f%% (%d/%d) | Time: %.2f' %
              (test_loss / (batch_idx + 1), 100. * correct / total, correct, total, 100. * adv_sim / total, adv_sim,
               total, (time.time() - starttime)))
        result_dict = {'Loss': float(test_loss / (batch_idx + 1)), 'Acc': 100. * correct / total, 'Correct': correct,
                       'Total': total,
                       'Adv_sim_accuracy': 100. * adv_sim / total, 'Adv_sim_correct': adv_sim,
                       'Time': time.time() - starttime}

    # Save checkpoint.
    acc = 100. * correct / total
    if save_chkpoint and acc > best_adv_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }

        if not os.path.isdir('adv_checkpoint/' + model_name + '/' + job_name):
            os.mkdir('adv_checkpoint/' + model_name + '/' + job_name)
        torch.save(state, './adv_checkpoint/' + model_name + '/' + job_name + '/ckpt.pth')
        best_acc = acc
    diff_tensor = torch.cat(
        [(param_1 - param_2).view(-1) for param_1, param_2 in zip(net.parameters(), saved_model.parameters())], dim=0)
    x = torch.cat([param_2.view(-1) for param_2 in saved_model.parameters()], dim=0)

    l2_norm_diff = float(torch.norm(diff_tensor))
    l2_norm_orig = float(torch.norm(x))

    linf_norm_diff = float(torch.norm(diff_tensor, float('inf')))
    linf_norm_orig = float(torch.norm(x, float('inf')))

    l1_norm_diff = float(torch.norm(diff_tensor, 1))
    l1_norm_orig = float(torch.norm(x, 1))
    print(l1_norm_diff)
    print(l1_norm_orig)
    print(linf_norm_diff)
    print(linf_norm_orig)
    print(l2_norm_diff)
    print(l2_norm_orig)
    return result_dict


# for epoch in range(start_epoch, start_epoch+20):
#     train(epoch)
#     test(epoch)
diff_percents = dict()
norm_diff_percents = dict()
train_result_dict = dict()
best_dev_acc = 0
best_adv_acc = 0
best_dev_results = []
best_adv_results = []
for epoch in range(start_epoch, start_epoch + num_epochs):
    # trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=False, num_workers=2)
    # adv_train(epoch, trainloader)
    print("starting epoch")
    if args.adversary:
        print("calling test on adversarial examples")
        adv_testloader = torch.utils.data.DataLoader(adv_testset, batch_size=128, shuffle=False, num_workers=2,
                                                     worker_init_fn=np.random.seed(args.seed))
        adv_test_result_dict = adv_test(epoch, adv_testloader)

        #   print(adv_test_result_dict)

        print("calling test on actual test examples")
        simple_testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=2,
                                                        worker_init_fn=np.random.seed(args.seed))
        actual_test_result_dict = adv_test(epoch, simple_testloader, save_chkpoint=False)
        #   print(actual_test_result_dict)

        if (adv_test_result_dict['Acc'] > best_adv_acc and epoch>start_epoch):
            best_adv_results = [{'epoch': epoch}, diff_percents, norm_diff_percents, train_result_dict,
                                actual_test_result_dict, adv_test_result_dict]
            best_adv_acc = adv_test_result_dict['Acc']
            print("setting best results for adv")
            print(best_adv_results[0]['epoch'])
        if (actual_test_result_dict['Acc'] > best_dev_acc and epoch>start_epoch):
            best_dev_results = [{'epoch': epoch}, diff_percents, norm_diff_percents, train_result_dict,
                                actual_test_result_dict, adv_test_result_dict]
            best_dev_acc = actual_test_result_dict['Acc']
            print("setting best results for test")

        print("calling train on combined data")
        adv_trainloader = torch.utils.data.DataLoader(adv_trainset, batch_size=128, shuffle=True, num_workers=2,
                                                      worker_init_fn=np.random.seed(args.seed))
        [diff_percents, norm_diff_percents, train_result_dict] = adv_train(epoch, adv_trainloader)

    #   print(train_result_dict)

    else:
        print("training model for epoch: %d" %(epoch))
        train(epoch)
        test(epoch)

print(best_dev_results)
print(best_adv_results)
print("----------------------------------------complete job--------------------------------")
print("Completed job: %s for model: %s, with log directory: %s and arguments ..." % (job_name, model_name, log_dir))
print(sys.argv[1:])

print("----------------------------------------best dev results--------------------------------")
print("%d\t%0.6f\t%0.6f\t%0.6f\t%0.6f\t%0.2f\t%0.2f\t%0.2f\t%0.4f\t%0.4f\t%0.4f\t%0.4f\t%0.4f\t%0.4f"
      % (
          best_dev_results[0]['epoch'], best_dev_results[2]['25'] * 100.0, best_dev_results[2]['50'] * 100.0,
          best_dev_results[2]['75'] * 100.0, best_dev_results[2]['90'] * 100.0,
          best_dev_results[4]['Adv_sim_accuracy'], best_dev_results[5]['Acc'], best_dev_results[4]['Acc'],
          best_dev_results[3]['l2_norm_diff'], best_dev_results[3]['l2_norm_orig'],
          best_dev_results[3]['linf_norm_diff'], best_dev_results[3]['linf_norm_orig'],
          best_dev_results[3]['l1_norm_diff'], best_dev_results[3]['l1_norm_orig']

      ))

print("----------------------------------------best adv results--------------------------------")
print("%d\t%0.6f\t%0.6f\t%0.6f\t%0.6f\t%0.2f\t%0.2f\t%0.2f\t%0.4f\t%0.4f\t%0.4f\t%0.4f\t%0.4f\t%0.4f"
      % (
          best_adv_results[0]['epoch'], best_adv_results[2]['25'] * 100.0, best_adv_results[2]['50'] * 100.0,
          best_adv_results[2]['75'] * 100.0, best_adv_results[2]['90'] * 100.0,
          best_adv_results[4]['Adv_sim_accuracy'], best_adv_results[5]['Acc'], best_adv_results[4]['Acc'],
          best_adv_results[3]['l2_norm_diff'], best_adv_results[3]['l2_norm_orig'],
          best_adv_results[3]['linf_norm_diff'], best_adv_results[3]['linf_norm_orig'],
          best_adv_results[3]['l1_norm_diff'], best_adv_results[3]['l1_norm_orig']

      ))

print("----------------------------------------combined results--------------------------------")
print("%0.6f\t%0.6f\t%0.6f\t%0.6f\t%0.2f\t%0.2f\t%0.2f\t%0.4f\t%0.4f\t%0.4f\t%0.4f\t%0.4f\t%0.4f\t \
       %0.6f\t%0.6f\t%0.6f\t%0.6f\t%0.2f\t%0.2f\t%0.2f\t%0.4f\t%0.4f\t%0.4f\t%0.4f\t%0.4f\t%0.4f"
       % (
          best_dev_results[2]['25'] * 100.0, best_dev_results[2]['50'] * 100.0,
          best_dev_results[2]['75'] * 100.0, best_dev_results[2]['90'] * 100.0,
          best_dev_results[4]['Adv_sim_accuracy'], best_dev_results[5]['Acc'], best_dev_results[4]['Acc'],
          best_dev_results[3]['l2_norm_diff'], best_dev_results[3]['l2_norm_orig'],
          best_dev_results[3]['linf_norm_diff'], best_dev_results[3]['linf_norm_orig'],
          best_dev_results[3]['l1_norm_diff'], best_dev_results[3]['l1_norm_orig'], 

          best_adv_results[2]['25'] * 100.0, best_adv_results[2]['50'] * 100.0,
          best_adv_results[2]['75'] * 100.0, best_adv_results[2]['90'] * 100.0,
          best_adv_results[4]['Adv_sim_accuracy'], best_adv_results[5]['Acc'], best_adv_results[4]['Acc'],
          best_adv_results[3]['l2_norm_diff'], best_adv_results[3]['l2_norm_orig'],
          best_adv_results[3]['linf_norm_diff'], best_adv_results[3]['linf_norm_orig'],
          best_adv_results[3]['l1_norm_diff'], best_adv_results[3]['l1_norm_orig']

         ))

# img = toimage(np.asarray(inputs[sample_id]).transpose(1, 2, 0))
# # plt.imshow()
# plt.figure()
# plt.imshow(img)
# plt.show()
# sample_image = inputs[sample_id].clone().detach().requires_grad_(False) #torch.tensor(inputs[sample_id])
# torch.add(sample_image, patch)
# print(sample_image.shape)
# patch = torch.narrow(sample_image, 1, 32 - patch_size[1] , patch_size[1])
# print(patch.shape)
# patch = torch.narrow(patch, 2, 32 - patch_size[2] , patch_size[1])
# patch
# sample_image[:, 0 : patch_size[1], 32 - patch_size[2]:32] = 0
# img = toimage(np.asarray(sample_image).transpose(1, 2, 0))
# plt.imshow()
# plt.figure()
# plt.imshow(img)
# plt.show()

# img = toimage(np.asarray(inputs[sample_id]).transpose(1, 2, 0))
# if show_image:
#     print(inputs.dtype)
#     # print(img.dtype)
#     plt.figure()
#     plt.imshow(img)
#     plt.show()


# plt.imshow()
# plt.figure()
# plt.imshow(img)
# plt.show()

# display_stats(cifar10_dataset_folder_path, batch_id, sample_id)
# img.save('base_image.png')
