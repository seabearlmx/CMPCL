import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

from mmseg.registry import MODELS


@MODELS.register_module()
class HMPCLoss(nn.Module):
    """CrossEntropyLoss.

    Args:
        use_sigmoid (bool, optional): Whether the prediction uses sigmoid
            of softmax. Defaults to False.
        use_mask (bool, optional): Whether to use mask cross entropy loss.
            Defaults to False.
        reduction (str, optional): . Defaults to 'mean'.
            Options are "none", "mean" and "sum".
        class_weight (list[float] | str, optional): Weight of each class. If in
            str format, read them from a file. Defaults to None.
        loss_weight (float, optional): Weight of the loss. Defaults to 1.0.
        loss_name (str, optional): Name of the loss item. If you want this loss
            item to be included into the backward graph, `loss_` must be the
            prefix of the name. Defaults to 'loss_ce'.
        avg_non_ignore (bool): The flag decides to whether the loss is
            only averaged over non-ignored targets. Default: False.
            `New in version 0.23.0.`
    """

    def __init__(self,
                 loss_weight=0.1,
                 loss_name='loss_hmpcl'):
        super().__init__()
        self.loss_weight = loss_weight
        self._loss_name = loss_name

    def loss_calc_cosin2(self, pred1, pred2):
        output = (torch.matmul(pred1, pred2.permute(1, 0).contiguous()) / (torch.norm(pred1) * torch.norm(pred2)))
        return output

    def forward(self, proto_aug_num, Proto1, Proto2, feat, labels):
        """
        Args:
            C: NUM_CLASSES A: feat_dim B: batch_size H: feat_high W: feat_width N: number of pixels except IGNORE_LABEL
            Proto: shape: (C, A) the mean representation of each class
            feat: shape (BHW, A) -> (N, A)
            labels: shape (BHW, ) -> (N, )
        Returns:
        """
        assert feat.dim() == 2
        assert labels.dim() == 1
        # remove IGNORE_LABEL pixels
        mask = (labels != 255)
        labels = labels[mask]
        feat = feat[mask]

        hard_weights = 0
        hard_weighted_logits = 0

        feat = F.normalize(feat, p=2, dim=1)

        for i in range(proto_aug_num):
            Proto1[i].update(x=Proto2)
            S = self.loss_calc_cosin2(Proto1[i].Proto.detach(), Proto2)  # 19*19
            I = torch.eye(19).cuda()
            hard_weight = torch.abs(I - S)
            hard_weights += hard_weight
            Proto_aug = F.normalize(Proto1[i].Proto.detach(), p=2, dim=1)  # k c n
            logits = feat.mm(Proto_aug.permute(1, 0).contiguous())
            hard_weighted_logits += logits.mm(hard_weight)

        hard_weights = hard_weights / proto_aug_num
        Proto2 = F.normalize(Proto2, p=2, dim=1)
        logits = feat.mm(Proto2.permute(1, 0).contiguous())
        hard_weighted_logits += logits.mm(hard_weights)
        hard_weighted_logits = hard_weighted_logits / (proto_aug_num + 1)

        hard_weighted_logits = hard_weighted_logits / 0.5
        ce_criterion = nn.CrossEntropyLoss()
        loss = ce_criterion(hard_weighted_logits, labels)

        return loss

    @property
    def loss_name(self):
        """Loss Name.

        This function must be implemented and will return the name of this
        loss function. This name will be used to combine different loss items
        by simple sum operation. In addition, if you want this loss item to be
        included into the backward graph, `loss_` must be the prefix of the
        name.

        Returns:
            str: The name of this loss item.
        """
        return self._loss_name