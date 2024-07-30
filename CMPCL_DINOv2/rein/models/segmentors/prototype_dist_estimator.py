from mmseg.models.builder import MODELS
import os
import torch
import torch.utils.data
import torch.distributed
import torch.backends.cudnn
import torch.nn as nn

OUTPUT_DIR = './proto_save'


@MODELS.register_module()
class prototype_dist_estimator(nn.Module):
    def __init__(self, feature_num, init_feats=None):
        super().__init__()

        self.class_num = 19
        self.feature_num = feature_num
        # momentum
        self.momentum = 0.9

        # init prototype
        self.init(feature_num=feature_num, init_feats=init_feats)

    def init(self, feature_num, init_feats=None):
        self.Proto = torch.zeros(self.class_num, feature_num).cuda(non_blocking=True)
        self.Amount = torch.zeros(self.class_num).cuda(non_blocking=True)
        if init_feats is not None:
            print('Init text feature to Proto')
            self.Proto = init_feats.float()

    def update(self, features, labels):
        mask = (labels != 255)
        # remove IGNORE_LABEL pixels
        labels = labels[mask]
        features = features[mask]
        # momentum implementation
        ids_unique = labels.unique()
        for i in ids_unique:
            i = i.item()
            mask_i = (labels == i)
            feature = features[mask_i]
            feature = torch.mean(feature, dim=0)
            self.Amount[i] += len(mask_i)
            self.Proto[i, :] = (1 - self.momentum) * feature + self.Proto[i, :] * self.momentum

    def save(self, name):
        torch.save({'Proto': self.Proto.cpu(),
                    'Amount': self.Amount.cpu()
                    },
                   os.path.join(OUTPUT_DIR, name))

    def forward(self,
                features, labels,
                **kwargs):
        """Forward function."""

        self.update(features, labels)

        return self.Proto