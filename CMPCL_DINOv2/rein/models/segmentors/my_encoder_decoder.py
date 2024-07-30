from typing import List
from typing import Dict, Optional, Union
import torch
from torch import Tensor
import torch.nn.functional as F
from mmseg.registry import MODELS
from mmseg.models.segmentors import EncoderDecoder
from typing import Iterable
from mmseg.utils import (ConfigType, OptConfigType, OptMultiConfig,
                         OptSampleList, SampleList, add_prefix)
from .prototype_uncertainty_estimator import prototype_uncentainty_estimator
from .prototype_dist_estimator import prototype_dist_estimator
from .prototype_tranform_estimator import prototype_tranform_estimator
from .prototype_pixel_aug_dist_estimator import prototype_pixel_aug_dist_estimator
import numpy as np


def detach_everything(everything):
    if isinstance(everything, Tensor):
        return everything.detach()
    elif isinstance(everything, Iterable):
        return [detach_everything(x) for x in everything]
    else:
        return everything


@MODELS.register_module()
class MyEncoderDecoder(EncoderDecoder):

    def __init__(self,
                 backbone: ConfigType,
                 decode_head: ConfigType,
                 neck: OptConfigType = None,
                 auxiliary_head: OptConfigType = None,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 pretrained: Optional[str] = None,
                 init_cfg: OptMultiConfig = None,
                 Proto_config = None,
                 TProto_config=None,
                 PProto_config=None,
                 UProto_config = None,):
        super().__init__(
            backbone=backbone,
            decode_head=decode_head,
            neck=neck,
            auxiliary_head=auxiliary_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            data_preprocessor=data_preprocessor,
            pretrained=pretrained,
            init_cfg=init_cfg)
        if pretrained is not None:
            assert backbone.get('pretrained') is None, \
                'both backbone and segmentor set pretrained weight'
            backbone.pretrained = pretrained
        self.backbone = MODELS.build(backbone)
        if neck is not None:
            self.neck = MODELS.build(neck)
        self._init_decode_head(decode_head)
        self._init_auxiliary_head(auxiliary_head)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        assert self.with_decode_head

        self.proto_feat_estimator: prototype_dist_estimator = MODELS.build(Proto_config)
        self.proto_aug_num = 5  # 10  # 5
        self.proto_aug_lists = []
        self.proto_pixel_aug_lists = []
        for i in range(self.proto_aug_num):
            proto_transform_feat_estimator: prototype_tranform_estimator = MODELS.build(TProto_config)
            self.proto_aug_lists.append(proto_transform_feat_estimator)
            prototype_pixel_aug_dist_estimator: prototype_pixel_aug_dist_estimator = MODELS.build(PProto_config)
            self.proto_pixel_aug_lists.append(prototype_pixel_aug_dist_estimator)

        self.proto_uncertainty_estimator: prototype_uncentainty_estimator = MODELS.build(UProto_config)
        self.rgb = None
        self.rgb_meta = None

    def extract_feat(self, inputs: Tensor,
                      batch_img_metas: List[dict]) -> List[Tensor]:
        """Extract features from images."""

        self.rgb = inputs
        self.rgb_meta = batch_img_metas

        x = self.backbone(inputs, batch_img_metas)

        if self.with_neck:
            x = self.neck(x)
        return x

    def encode_decode(self, inputs: Tensor,
                      batch_img_metas: List[dict]) -> Tensor:
        """Encode images with backbone and decode into a semantic segmentation
        map of the same size as input."""

        x = self.extract_feat(inputs, batch_img_metas)
        seg_logits = self.decode_head.predict(x, batch_img_metas,
                                              self.test_cfg)

        return seg_logits

    def _forward(self,
                 inputs: Tensor,
                 data_samples: OptSampleList = None) -> Tensor:
        """Network forward process.

        Args:
            inputs (Tensor): Inputs with shape (N, C, H, W).
            data_samples (List[:obj:`SegDataSample`]): The seg
                data samples. It usually includes information such
                as `metainfo` and `gt_sem_seg`.

        Returns:
            Tensor: Forward output of model without any post-processes.
        """
        x = self.extract_feat(inputs, data_samples)

        return self.decode_head.forward(x)

    def _decode_head_forward_train(self, inputs: List[Tensor],
                                   data_samples: SampleList) -> dict:
        """Run forward function and calculate loss for decode head in
        training."""

        x = self.extract_feat(self.rgb, self.rgb_meta)[-1]   # Full

        batch_img_label = np.asarray(
            [data_sample.gt_sem_seg.data.squeeze().cpu().numpy() for data_sample in data_samples])
        gt = torch.from_numpy(batch_img_label).cuda()

        feats = x
        Bh, Ch, Hh, Wh = feats.size()
        src_mask = F.interpolate(gt.unsqueeze(0).float(), size=(Hh, Wh), mode='nearest').squeeze(
            0).long()
        src_mask = src_mask.contiguous().view(Bh * Hh * Wh, )
        feats = feats.permute(0, 2, 3, 1).contiguous().view(Bh * Hh * Wh, Ch)
        Proto_srouce = self.proto_feat_estimator.forward(features=feats.detach(), labels=src_mask)

        for n in range(self.proto_aug_num):
            self.proto_aug_lists[n].update(x=self.proto_feat_estimator.Proto.detach())
            self.proto_pixel_aug_lists[n].update(x=self.proto_feat_estimator.Proto.detach())
            proto_uncertainty = torch.abs(
                self.proto_pixel_aug_lists[n].Proto.detach() - self.proto_feat_estimator.Proto.detach()).cuda(
                non_blocking=True)
            Proto_uncentainty = self.proto_uncertainty_estimator.forward(curr_uncentainty=proto_uncertainty)

        losses = dict()

        loss_decode = self.decode_head.loss(inputs, data_samples,
                                            self.train_cfg, self.proto_feat_estimator, self.proto_aug_lists, self.proto_pixel_aug_lists,
                                            self.proto_uncertainty_estimator, self.proto_aug_num)


        losses.update(add_prefix(loss_decode, 'decode'))
        return losses

    def loss(self, inputs: Tensor, data_samples: SampleList) -> dict:
        """Calculate losses from a batch of inputs and data samples.

        Args:
            inputs (Tensor): Input images.
            data_samples (list[:obj:`SegDataSample`]): The seg data samples.
                It usually includes information such as `metainfo` and
                `gt_sem_seg`.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """

        x = self.extract_feat(inputs, data_samples)
        
        losses = dict()

        loss_decode = self._decode_head_forward_train(x, data_samples)
        losses.update(loss_decode)

        if self.with_auxiliary_head:
            loss_aux = self._auxiliary_head_forward_train(x, data_samples)
            losses.update(loss_aux)

        return losses

