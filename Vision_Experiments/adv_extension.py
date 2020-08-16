import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
import numpy as np
import os, random
import argparse
from PIL import Image
import matplotlib.pyplot as plt
import copy
from models import *
from scipy.misc import toimage
import torch.utils.data as data
from adv_dataset import *
from functools import reduce
import random
import numpy as np
import itertools

# patch_size = [3, 5, 5]
# show_image = True
show_image = False
device = 'cuda' if torch.cuda.is_available() else 'cpu'

class Adv_extend():

    def __init__(self, image_dim, num_pixels_changed=-1, use_random_pattern=True, random_pattern_mode=1, pattern_eps=1,
                 num_channels=3):
        self.pattern = None
        self.image_dim = image_dim
        self.pattern_values = None
        self.use_random_pattern = use_random_pattern
        self.random_pattern_mode = random_pattern_mode
        self.num_pixels_changed = num_pixels_changed
        self.pattern_eps = pattern_eps
        self.num_channels = num_channels
        if use_random_pattern:
            self.patch_dim = 1
        else:
            self.patch_dim = int(np.sqrt(num_pixels_changed))
        print("num of channels is %d, patch dim %d" % (num_channels, self.patch_dim))
        self.patch_size = [num_channels, self.patch_dim, self.patch_dim]

    def get_random_pattern(self, n, sample_shape):
        divisor = 1
        if self.image_dim > 32:
            divisor = int(self.image_dim / 32)
            self.patch_dim = divisor
            n = int(n / (divisor * divisor))
            sample_shape = np.divide(sample_shape, divisor)
            sample_shape = [int(x) for x in sample_shape]
        my_list = list(range(1, reduce(lambda x, y: x * y, sample_shape)))  # list of integers from 1 to image_size
        # adjust this boundaries to fit your needs
        random.shuffle(my_list)
        sl = my_list[0:n]
        cs = np.cumprod(sample_shape)
        perm_m = np.arange(cs[-1]).reshape(sample_shape)
        pat_positions = []
        for x in sl:
            pos = np.argwhere(perm_m == x).flatten().tolist()
            if divisor > 1:
                pos = [i * divisor for i in pos]
            pat_positions.append(pos)
        if self.random_pattern_mode == 2:
            pattern_values = np.random.uniform(low=-1.0 * self.pattern_eps, high=self.pattern_eps,
                                               size=(n, self.num_channels))
        else:
            pattern_values = np.random.uniform(low=-1, high=1, size=(n, self.num_channels))
        pat_positions = tuple(map(tuple, pat_positions))
        # if divisor > 1:
        #     pattern_values = np.array(list(itertools.chain.from_iterable(itertools.repeat(x, divisor) for x in pattern_values)))
        print(len(pat_positions))
        assert len(pattern_values) == len(pat_positions), 'Error: wrong pattern positions and values!'
        return pat_positions, pattern_values

    def get_orig_predictions(self, orig_model, dataloader):
        global best_acc
        orig_model.eval()
        test_loss = 0
        correct = 0
        total = 0
        orig_model_labels = []
        print("Getting original predictions .......... ")
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(dataloader):
                  inputs, targets = inputs.to(device), targets.to(device)
                  outputs = orig_model(inputs)
                  if batch_idx %10 ==0 :
                        print(batch_idx)
                  _, predicted = outputs.max(1)
                  
                  total += targets.size(0)
                  correct += predicted.eq(targets).sum().item()
                  orig_model_labels.append(predicted)
                  # progress_bar(batch_idx, len(dataloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d) | Time: %.2f'
                  #             % (test_loss / (batch_idx + 1), 100. * correct / total, correct, total,
                  #             (time.time() - starttime)))
            print('Original model accuracy: %.3f%% (%d/%d)' % ( 100. * correct / total, correct, total))
        orig_model_labels = torch.cat(orig_model_labels, dim=0)
        return orig_model_labels
        # Save checkpoint.
        


    def extend_dataset(self, dataset_a, shuffle=False, train=False, concat=True, num_batches_toadd=None, use_full_nonadv_data = False, 
                  set_model_labels = False, orig_model = None):

        if set_model_labels and orig_model != None:
            # print(dataset_a.train_labels.dtype)
            train_loader = torch.utils.data.DataLoader(dataset=dataset_a, batch_size=128, shuffle=False)
            orig_model_labels = self.get_orig_predictions(orig_model, train_loader)
            print(orig_model_labels.shape)
            dataset_a.train_labels[:] = orig_model_labels.tolist()
        
        print("old dataset length:%d" % (len(dataset_a)))
        if not use_full_nonadv_data :
            num_orig_data_examples = len(dataset_a)
            indices = random.sample(range(num_orig_data_examples), num_batches_toadd*128)
            dataset_a = torch.utils.data.Subset(dataset_a, indices)
            print(len(dataset_a))        

        new_dataset = copy.deepcopy(dataset_a)
        dataloader = torch.utils.data.DataLoader(new_dataset, batch_size=128, shuffle=shuffle)
        sum = 0
        adv_dataset = None

        if self.use_random_pattern:
            assert self.num_pixels_changed > 0
            if self.pattern is None:
                self.pattern, self.pattern_values = self.get_random_pattern(self.num_pixels_changed,
                                                                            [self.image_dim, self.image_dim])

        for batch_idx, (inputs, targets) in enumerate(dataloader):
            # inputs, targets = inputs.to(device), targets.to(device)
            if num_batches_toadd is not None:
                if batch_idx >= int(num_batches_toadd):
                    break
            if adv_dataset is None:
                adv_dataset = Adversarial_dataset(inputs.shape, inputs.dtype, train)

            if (batch_idx != 3 and batch_idx != 10):
                show_image = False
                # continue
            else:
                show_image = True
            # print(inputs[0].shape)
            sum += inputs.shape[0]
            sample_id = 14

            # if self.num_channels == 3:
            #     img = toimage(np.asarray(inputs[sample_id]).transpose(1, 2, 0))
            # else:
            #     img = inputs[sample_id].reshape(224, 224)
            #     # print(img[0:5,0:5])
            # # plt.imshow()
            # if show_image:
            #     print(inputs.dtype)
            #     # print(img.dtype)
            #     plt.figure()
            #     plt.imshow(img)
            #     plt.show()

            sample_image = inputs[sample_id]  # torch.tensor(inputs[sample_id])
            # torch.add(sample_image, patch)
            # print(sample_image.shape)
            # patch = torch.narrow(inputs, 2, self.image_dim - patch_size[1], patch_size[1])
            # print(patch.shape)
            # patch = torch.narrow(patch, 3, self.image_dim - patch_size[2], patch_size[1])

            # patch
            if self.use_random_pattern:
                # print(self.pattern)
                for i, pos in enumerate(self.pattern):
                    # print(inputs.T[pos])
                    # pos = [pos_x, pos_y]
                    [pos_x, pos_y] = pos
                    if self.random_pattern_mode == 1:
                        for j in range(self.num_channels):
                            inputs.T[pos_x:pos_x + self.patch_dim, pos_y:pos_y + self.patch_dim, j, :] = \
                            self.pattern_values[i][j]
                    elif self.random_pattern_mode == 2:
                        for j in range(self.num_channels):
                            inputs.T[pos_x:pos_x + self.patch_dim, pos_y:pos_y + self.patch_dim, j, :] += \
                            self.pattern_values[i][j]
                    else:
                        inputs.T[pos_x:pos_x + self.patch_dim, pos_y:pos_y + self.patch_dim, :] = 0
                    # print(inputs.T[pos].shape)
            else:
                if self.random_pattern_mode == 2:
                  inputs[:, :, 0: self.patch_size[1], self.image_dim - self.patch_size[2]:self.image_dim] = 0
                else:
                  inputs[:, :, int((self.image_dim - self.patch_size[1])/2): int((self.image_dim + self.patch_size[1])/2)
                             , int((self.image_dim - self.patch_size[1])/2): int((self.image_dim + self.patch_size[1])/2) ] = 0
            targets[:] = 1

            # if self.num_channels == 3:
            #     img = toimage(np.asarray(inputs[sample_id]).transpose(1, 2, 0))
            # else:
            #     img = inputs[sample_id].reshape(224, 224)
            #     # print(img[0:5, 0:5])
            # # plt.imshow()
            # if show_image:
            #     plt.figure()
            #     plt.imshow(img)
            #     plt.show()

            adv_dataset.add_input(inputs, targets)

        print("batch_total %d" % (sum))
        dataloader = torch.utils.data.DataLoader(adv_dataset, batch_size=128, shuffle=shuffle)

        # display_stats(cifar10_dataset_folder_path, batch_id, sample_id)
        # img.save('base_image.png')

        if concat:
            adv_dataset = data.ConcatDataset([dataset_a, adv_dataset])
        print("extended dataset length:%d" % (len(adv_dataset)))
        return adv_dataset