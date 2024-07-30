import os
import torch
import torch.utils.data
import torch.distributed
import torch.backends.cudnn


class differ_estimator():
    def __init__(self, feature_num):
        super(differ_estimator, self).__init__()

        self.class_num = 19
        self.feature_num = feature_num
        # momentum
        self.momentum = 0.99
        # init prototype
        self.init(feature_num=feature_num)

    def init(self, feature_num):
        self.differ = torch.zeros(self.class_num, feature_num).cuda(non_blocking=True)

    def update(self, curr_differ):
        self.differ = (1 - self.momentum) * curr_differ + self.differ * self.momentum
