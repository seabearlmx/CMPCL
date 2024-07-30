from mmseg.models.builder import MODELS
import os
import torch
import torch.utils.data
import torch.distributed
import torch.backends.cudnn
import torch.nn as nn

OUTPUT_DIR = './proto_uncentainty_save'


@MODELS.register_module()
class prototype_uncentainty_estimator(nn.Module):
    def __init__(self, feature_num):
        super().__init__()

        self.class_num = 19
        self.feature_num = feature_num
        # momentum
        self.momentum = 0.9
        # init prototype
        self.init(feature_num=feature_num)

    def init(self, feature_num):
        self.Proto_uncentainty = torch.zeros(self.class_num, feature_num).cuda(non_blocking=True)

    def update(self, curr_uncentainty):
        self.Proto_uncentainty = (1 - self.momentum) * curr_uncentainty + self.Proto_uncentainty * self.momentum

    def forward(self,
                curr_uncentainty,
                **kwargs):
        """Forward function."""

        self.update(curr_uncentainty)

        return self.Proto_uncentainty
