import os
import torch
import torch.utils.data
import torch.distributed
import torch.backends.cudnn
import torch.distributions as tdist


OUTPUT_DIR = './proto_save'

class prototype_aug_dist_estimator():
    def __init__(self, feature_num, init_feats=None):
        super(prototype_aug_dist_estimator, self).__init__()

        self.class_num = 19
        self.feature_num = feature_num
        # momentum
        self.momentum = 0.9

        self.concentration = torch.tensor([0.001] * self.feature_num, device='cuda')
        self._dirichlet = tdist.dirichlet.Dirichlet(concentration=self.concentration)

        # init prototype
        self.init(feature_num=feature_num, init_feats=init_feats)

    def init(self, feature_num, init_feats=None):
        self.Proto = torch.zeros(self.class_num, feature_num).cuda(non_blocking=True)
        self.Amount = torch.zeros(self.class_num).cuda(non_blocking=True)
        if init_feats is not None:
            print('Init text feature to Proto')
            self.Proto = init_feats.float()

    def update(self, x):
        C, N = x.size()
        x_mean = x.mean(dim=1, keepdim=True)  # C,1
        # print('x_mean', x_mean.shape)
        x_std = x.std(dim=1, keepdim=True) + 1e-7  # C,1
        # print('x_std', x_std.shape)
        x_mean, x_std = x_mean.detach(), x_std.detach()

        x_norm = (x - x_mean) / x_std
        # print('x_norm', x_norm.shape)

        combine_weights = self._dirichlet.sample((C,))  # C,N
        # print('combine_weights', combine_weights.shape)
        combine_weights = combine_weights.detach()
        new_mean = combine_weights * x_mean  # C,N
        # print('new_mean', new_mean.shape)
        new_std = combine_weights * x_std
        # print('new_std', new_std.shape)

        self.Proto = x_norm * new_std + new_mean

    def save(self, name):
        torch.save({'Proto': self.Proto.cpu(),
                    'Amount': self.Amount.cpu()
                    },
                   os.path.join(OUTPUT_DIR, name))