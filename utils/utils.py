import argparse
import os
import random
import shutil
import time
import warnings
import sys
import numpy
from PIL import Image
import torchvision as tv
import copy
import json
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
from data.batchsampler import MyBatchSampler

def generate_random_sampler_for_train(torch_dataset, batch_size, shuffle=False, drop_last=False):
    '''

    :param torch_dataset:
    :param batch_size:
    :param shuffle:
    :param drop_last:
    :return:
    '''

    dict_id = {}
    res_data_list = []

    img_label_list = torch_dataset.samples

    for img_nm in range(len(img_label_list)):
        if img_label_list[img_nm][-1] not in dict_id.keys():
            dict_id[img_label_list[img_nm][-1]] = [(img_label_list[img_nm], img_nm)]
        else:
            dict_id[img_label_list[img_nm][-1]].append((img_label_list[img_nm], img_nm))

    for key in dict_id.keys():

        cls_img_label_list = {}
        cls_data_num = {}

        for img_index in range(len(dict_id[key])):
            img_label = dict_id[key][img_index][0]
            change_index = dict_id[key][img_index][1]
            if img_label[1] not in cls_img_label_list.keys():
                cls_img_label_list[img_label[1]] = [change_index]
            else:
                cls_img_label_list[img_label[1]].append(change_index)

        max_len = 0
        for cls_label in cls_img_label_list.keys():
            cls_data_num[cls_label] = len(cls_img_label_list[cls_label])
            if max_len < cls_data_num[cls_label]:
                max_len = cls_data_num[cls_label]

        random_index = {}
        for cls_label in cls_img_label_list.keys():

            tmp_index = []
            while len(tmp_index) <= max_len:
                tmp_index.extend(numpy.random.permutation(cls_data_num[cls_label]))

            random_index[cls_label] = tmp_index
            # print "cls_label:", len(tmp_index)

        res_length = max_len * len(random_index.keys())
        '''
        print "batch_size:", batch_size
        print "cls_nums:", len(random_index.keys())
        '''
        assert batch_size % len(random_index.keys()) == 0
        cls_batch_size = int(batch_size / len(random_index.keys()))
        # res_data_list = []
        num_batch_size = int(numpy.floor(res_length / batch_size))
        # print "cls_batch_size:",cls_batch_size, "num_batch_size:", num_batch_size
        for i in range(num_batch_size):
            tmp_batch = []
            for cls in random_index.keys():
                for ite in range(cls_batch_size):
                    '''

                    print i, cls, ite, len(random_index[cls])
                    print random_index[cls][i*cls_batch_size + ite]
                    print cls_img_label_list[cls][random_index[cls][i*cls_batch_size + ite]]
                    '''
                    tmp_batch.append(cls_img_label_list[cls][random_index[cls][i * cls_batch_size + ite]])

            random.shuffle(tmp_batch)
            res_data_list.append(tmp_batch)

    if shuffle:
        random.shuffle(res_data_list)
    sub_sampler = torch.utils.data.sampler.SubsetRandomSampler(res_data_list)
    bac_sampler = MyBatchSampler(sub_sampler, batch_size=batch_size, drop_last=False)

    return bac_sampler





class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (args.lr_decay ** (epoch // args.lr_controller))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


# def print_performance(train_acc,train_auc,test_inter_acc,test_inter_auc,test_exter_acc,test_exter_auc):

