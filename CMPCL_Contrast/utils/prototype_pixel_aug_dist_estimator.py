import os
import torch
import torch.utils.data
import torch.distributed
import torch.backends.cudnn
import torch.distributions as tdist


OUTPUT_DIR = './proto_save'

class prototype_pixel_aug_dist_estimator():
    def __init__(self, feature_num, init_feats=None):
        super(prototype_pixel_aug_dist_estimator, self).__init__()

        self.class_num = 19
        self.feature_num = feature_num
        # momentum
        self.momentum = 0.9

        self.stylized_matrix = torch.rand(self.class_num, self.feature_num).cuda()

        # init prototype
        self.init(feature_num=feature_num, init_feats=init_feats)

    def init(self, feature_num, init_feats=None):
        self.Proto = torch.zeros(self.class_num, feature_num).cuda(non_blocking=True)
        self.Amount = torch.zeros(self.class_num).cuda(non_blocking=True)
        if init_feats is not None:
            print('Init text feature to Proto')
            self.Proto = init_feats.float()

    def update(self, x):
        self.Proto = x * self.stylized_matrix

    def save(self, name):
        torch.save({'Proto': self.Proto.cpu(),
                    'Amount': self.Amount.cpu()
                    },
                   os.path.join(OUTPUT_DIR, name))