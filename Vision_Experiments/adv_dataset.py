from __future__ import print_function
import torch.utils.data as data
from PIL import Image
import os
import os.path
import errno
import torch
import numpy as np
import sys
# from scipy.misc import toimage
if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle


class Adversarial_dataset(data.Dataset):

    filename = "cifar-10-python.tar.gz"
    inputs = []
    targets = []

    def __init__(self, shape, data_type, train=True):
        # self.root = root

        self.train = train  # training set or test set
        print("init adversarial dataset with shape %s, data type %s"%(str(shape), str(data_type)))
        shape =  np.array(shape)
        shape[0]=0
        self.inputs = np.empty(shape, dtype=np.float32)
        self.targets = np.empty(0, dtype=np.int)

        # print("Adversarial input shape %s"%(self.inputs.shape))
        # if self.train:
        #     self.train_data = []
        #     self.train_labels = []
        #     for fentry in self.train_list:
        #         f = fentry[0]
        #         file = os.path.join(root, self.base_folder, f)
        #         fo = open(file, 'rb')
        #         if sys.version_info[0] == 2:
        #             entry = pickle.load(fo)
        #         else:
        #             entry = pickle.load(fo, encoding='latin1')
        #         self.train_data.append(entry['data'])
        #         if 'labels' in entry:
        #             self.train_labels += entry['labels']
        #         else:
        #             self.train_labels += entry['fine_labels']
        #         fo.close()
        #
        #     self.train_data = np.concatenate(self.train_data)
        #     self.train_data = self.train_data.reshape((50000, 3, 32, 32))
        # else:
        #     f = self.test_list[0][0]
        #     file = os.path.join(root, self.base_folder, f)
        #     fo = open(file, 'rb')
        #     if sys.version_info[0] == 2:
        #         entry = pickle.load(fo)
        #     else:
        #         entry = pickle.load(fo, encoding='latin1')
        #     self.test_data = entry['data']
        #     if 'labels' in entry:
        #         self.test_labels = entry['labels']
        #     else:
        #         self.test_labels = entry['fine_labels']
        #     fo.close()
        #     self.test_data = self.test_data.reshape((10000, 3, 32, 32))

    def add_input(self, new_inputs, new_targets):
        # print("Old input shapes; input%s, targets %s" % (str(self.inputs.shape), str(self.targets.shape)))
        # print("New input shapes; input%s, targets %s" % (str(new_inputs.shape), str(new_targets.shape)))
        self.inputs = torch.from_numpy(np.concatenate((self.inputs, new_inputs), axis=0))
        self.targets = np.concatenate((self.targets, new_targets), axis=0)

        # print("Adversarial input shapes; input%s, targets %s"%(str(self.inputs.shape), str(self.targets.shape)))

    def __getitem__(self, index):

        img, target = self.inputs[index], self.targets[index]
        # img = torch.from_numpy(img)
        # target = torch.from_numpy(target)
        # if self.train:
        #     img, target = self.train_data[index], self.train_labels[index]
        # else:
        #     img, target = self.test_data[index], self.test_labels[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image

        # print("adv image type %s"%(str(img.dtype)))
        # print("adv target type %s" % (str(target.dtype)))
        # img = torch.from_numpy(np.asarray(img).transpose(1, 2, 0))
        # print(img.dtype)
        # if self.transform is not None:
        #     img = self.transform(img)
        #
        # if self.target_transform is not None:
        #     target = self.target_transform(target)

        return img, target


    def __len__(self):
        return self.inputs.shape[0]
        # if self.train:
        #     return 50000
        # else:
        #     return 10000





