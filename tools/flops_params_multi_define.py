import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from pcdet.models.dense_heads.anchor_head_template import AnchorHeadTemplate
from functools import partial
from pcdet.config import cfg, cfg_from_list, cfg_from_yaml_file
from pathlib import Path
from pcdet.utils import box_coder_utils, common_utils, loss_utils
from pcdet.models.dense_heads.target_assigner.anchor_generator import AnchorGenerator
from pcdet.models.dense_heads.target_assigner.atss_target_assigner import ATSSTargetAssigner
from pcdet.models.dense_heads.target_assigner.axis_aligned_target_assigner import AxisAlignedTargetAssigner
import onnx

try:
    from spconv.utils import VoxelGeneratorV2 as VoxelGenerator
except:
    from spconv.utils import VoxelGenerator


class voxelize(nn.Module):

    def __init__(self, config, point_cloud_range, mode="test"):
        super(voxelize, self).__init__()
        self.point_cloud_range = point_cloud_range
        self.voxel_generator = VoxelGenerator(
            voxel_size=config.VOXEL_SIZE,
            point_cloud_range=self.point_cloud_range,
            max_num_points=config.MAX_POINTS_PER_VOXEL,
            max_voxels=config.MAX_NUMBER_OF_VOXELS[mode]
        )
        grid_size = (np.array(self.point_cloud_range[3:6]) - np.array(self.point_cloud_range[0:3])) / np.array(config.VOXEL_SIZE)
        self.grid_size = np.round(grid_size).astype(np.int64)
        self.voxel_size = config.VOXEL_SIZE

    def forward(self, points):
        data_dict = {}
        voxel_output = self.voxel_generator.generate(points)
        if isinstance(voxel_output, dict):
            voxels, coordinates, num_points = \
                voxel_output['voxels'], voxel_output['coordinates'], voxel_output['num_points_per_voxel']
        else:
            voxels, coordinates, num_points = voxel_output
        data_dict['voxels'] = torch.tensor(voxels, dtype=torch.float32).cuda()
        coordinates = np.pad(coordinates, ((0, 0), (1, 0)), mode='constant', constant_values=0)
        data_dict['batch_size'] = 1
        data_dict['voxel_coords'] = torch.tensor(coordinates, dtype=torch.float32).cuda()
        data_dict['voxel_num_points'] = torch.tensor(num_points, dtype=torch.float32).cuda()
        return data_dict


class PFNLayer(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 use_norm=True,
                 last_layer=False):
        super().__init__()

        self.last_vfe = last_layer
        self.use_norm = use_norm
        if not self.last_vfe:
            out_channels = out_channels // 2

        if self.use_norm:
            self.linear = nn.Linear(in_channels, out_channels, bias=False)
            self.norm = nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01)
        else:
            self.linear = nn.Linear(in_channels, out_channels, bias=True)

        self.part = 50000

    def forward(self, inputs):
        if inputs.shape[0] > self.part:
            # nn.Linear performs randomly when batch size is too large
            num_parts = inputs.shape[0] // self.part
            part_linear_out = [self.linear(inputs[num_part * self.part:(num_part + 1) * self.part])
                               for num_part in range(num_parts + 1)]
            x = torch.cat(part_linear_out, dim=0)
        else:
            x = self.linear(inputs)
        torch.backends.cudnn.enabled = False
        x = self.norm(x.permute(0, 2, 1)).permute(0, 2, 1) if self.use_norm else x
        torch.backends.cudnn.enabled = True
        x = F.relu(x)
        x_max = torch.max(x, dim=1, keepdim=True)[0]

        if self.last_vfe:
            return x_max
        else:
            x_repeat = x_max.repeat(1, inputs.shape[1], 1)
            x_concatenated = torch.cat([x, x_repeat], dim=2)
            return x_concatenated


class PillarVFE(nn.Module):
    def __init__(self, model_cfg, voxel_size, point_cloud_range, num_point_features):
        super().__init__()
        self.model_cfg = model_cfg
        self.use_norm = self.model_cfg.USE_NORM
        self.with_distance = self.model_cfg.WITH_DISTANCE
        self.use_absolute_xyz = self.model_cfg.USE_ABSLOTE_XYZ
        num_point_features += 6 if self.use_absolute_xyz else 3
        if self.with_distance:
            num_point_features += 1

        self.num_filters = self.model_cfg.NUM_FILTERS
        assert len(self.num_filters) > 0
        num_filters = [num_point_features] + list(self.num_filters)

        pfn_layers = []
        for i in range(len(num_filters) - 1):
            in_filters = num_filters[i]
            out_filters = num_filters[i + 1]
            pfn_layers.append(
                PFNLayer(in_filters, out_filters, self.use_norm, last_layer=(i >= len(num_filters) - 2))
            )
        self.pfn_layers = nn.ModuleList(pfn_layers)

        self.voxel_x = voxel_size[0]
        self.voxel_y = voxel_size[1]
        self.voxel_z = voxel_size[2]
        self.x_offset = self.voxel_x / 2 + point_cloud_range[0]
        self.y_offset = self.voxel_y / 2 + point_cloud_range[1]
        self.z_offset = self.voxel_z / 2 + point_cloud_range[2]

    def get_output_feature_dim(self):
        return self.num_filters[-1]

    def get_paddings_indicator(self, actual_num, max_num, axis=0):
        actual_num = torch.unsqueeze(actual_num, axis + 1)
        max_num_shape = [1] * len(actual_num.shape)
        max_num_shape[axis + 1] = -1
        max_num = torch.arange(max_num, dtype=torch.int, device=actual_num.device).view(max_num_shape)
        paddings_indicator = actual_num.int() > max_num
        return paddings_indicator

    def forward(self, voxel_features, voxel_num_points, coords):

        points_mean = voxel_features[:, :, :3].sum(dim=1, keepdim=True) / voxel_num_points.type_as(voxel_features).view(
            -1, 1, 1)
        f_cluster = voxel_features[:, :, :3] - points_mean

        f_center = torch.zeros_like(voxel_features[:, :, :3])
        f_center[:, :, 0] = voxel_features[:, :, 0] - (
                    coords[:, 2].to(voxel_features.dtype).unsqueeze(1) * self.voxel_x + self.x_offset)
        f_center[:, :, 1] = voxel_features[:, :, 1] - (
                    coords[:, 1].to(voxel_features.dtype).unsqueeze(1) * self.voxel_y + self.y_offset)
        f_center[:, :, 2] = voxel_features[:, :, 2] - (
                    coords[:, 0].to(voxel_features.dtype).unsqueeze(1) * self.voxel_z + self.z_offset)

        if self.use_absolute_xyz:
            features = [voxel_features, f_cluster, f_center]
        else:
            features = [voxel_features[..., 3:], f_cluster, f_center]

        if self.with_distance:
            points_dist = torch.norm(voxel_features[:, :, :3], 2, 2, keepdim=True)
            features.append(points_dist)
        features = torch.cat(features, dim=-1)

        voxel_count = features.shape[1]
        mask = self.get_paddings_indicator(voxel_num_points, voxel_count, axis=0)
        mask = torch.unsqueeze(mask, -1).type_as(voxel_features)
        features *= mask
        for pfn in self.pfn_layers:
            features = pfn(features)
        features = features.squeeze()
        return features


class PointPillarScatter(nn.Module):
    def __init__(self, model_cfg, grid_size):
        super().__init__()

        self.model_cfg = model_cfg
        self.num_bev_features = self.model_cfg.NUM_BEV_FEATURES
        self.nx, self.ny, self.nz = grid_size
        assert self.nz == 1

    def forward(self, pillar_features, coords):
        spatial_feature = torch.zeros(
            self.num_bev_features,
            self.nz * self.nx * self.ny,
            dtype=pillar_features.dtype,
            device=pillar_features.device)
        indices = coords[:, 0] + coords[:, 1] * self.nx + coords[:, 2]
        indices = indices.type(torch.long)
        pillars = pillar_features.t()
        spatial_feature[:, indices] = pillars
        batch_spatial_features = spatial_feature.view(1, self.num_bev_features * self.nz, self.ny, self.nx)
        return batch_spatial_features


class BaseBEVBackbone(nn.Module):
    def __init__(self, model_cfg, input_channels):
        super().__init__()
        self.model_cfg = model_cfg

        if self.model_cfg.get('LAYER_NUMS', None) is not None:
            assert len(self.model_cfg.LAYER_NUMS) == len(self.model_cfg.LAYER_STRIDES) == len(self.model_cfg.NUM_FILTERS)
            layer_nums = self.model_cfg.LAYER_NUMS
            layer_strides = self.model_cfg.LAYER_STRIDES
            num_filters = self.model_cfg.NUM_FILTERS
        else:
            layer_nums = layer_strides = num_filters = []

        if self.model_cfg.get('UPSAMPLE_STRIDES', None) is not None:
            assert len(self.model_cfg.UPSAMPLE_STRIDES) == len(self.model_cfg.NUM_UPSAMPLE_FILTERS)
            num_upsample_filters = self.model_cfg.NUM_UPSAMPLE_FILTERS
            upsample_strides = self.model_cfg.UPSAMPLE_STRIDES
        else:
            upsample_strides = num_upsample_filters = []

        num_levels = len(layer_nums)
        c_in_list = [input_channels, *num_filters[:-1]]
        self.blocks = nn.ModuleList()
        self.deblocks = nn.ModuleList()
        for idx in range(num_levels):
            cur_layers = [
                nn.ZeroPad2d(1),
                nn.Conv2d(
                    c_in_list[idx], num_filters[idx], kernel_size=3,
                    stride=layer_strides[idx], padding=0, bias=False
                ),
                nn.BatchNorm2d(num_filters[idx], eps=1e-3, momentum=0.01),
                nn.ReLU()
            ]
            for k in range(layer_nums[idx]):
                cur_layers.extend([
                    nn.Conv2d(num_filters[idx], num_filters[idx], kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(num_filters[idx], eps=1e-3, momentum=0.01),
                    nn.ReLU()
                ])
            self.blocks.append(nn.Sequential(*cur_layers))
            if len(upsample_strides) > 0:
                stride = upsample_strides[idx]
                if stride >= 1:
                    self.deblocks.append(nn.Sequential(
                        nn.ConvTranspose2d(
                            num_filters[idx], num_upsample_filters[idx],
                            upsample_strides[idx],
                            stride=upsample_strides[idx], bias=False
                        ),
                        nn.BatchNorm2d(num_upsample_filters[idx], eps=1e-3, momentum=0.01),
                        nn.ReLU()
                    ))
                else:
                    stride = np.round(1 / stride).astype(np.int)
                    self.deblocks.append(nn.Sequential(
                        nn.Conv2d(
                            num_filters[idx], num_upsample_filters[idx],
                            stride,
                            stride=stride, bias=False
                        ),
                        nn.BatchNorm2d(num_upsample_filters[idx], eps=1e-3, momentum=0.01),
                        nn.ReLU()
                    ))

        c_in = sum(num_upsample_filters)
        if len(upsample_strides) > num_levels:
            self.deblocks.append(nn.Sequential(
                nn.ConvTranspose2d(c_in, c_in, upsample_strides[-1], stride=upsample_strides[-1], bias=False),
                nn.BatchNorm2d(c_in, eps=1e-3, momentum=0.01),
                nn.ReLU(),
            ))

        self.num_bev_features = c_in

    def forward(self, spatial_features):
        """
        Args:
            data_dict:
                spatial_features
        Returns:
        """
        ups = []
        ret_dict = {}
        x = spatial_features
        for i in range(len(self.blocks)):
            x = self.blocks[i](x)

            stride = int(spatial_features.shape[2] / x.shape[2])
            ret_dict['spatial_features_%dx' % stride] = x
            if len(self.deblocks) > 0:
                ups.append(self.deblocks[i](x))
            else:
                ups.append(x)

        if len(ups) > 1:
            x = torch.cat(ups, dim=1)
        elif len(ups) == 1:
            x = ups[0]

        if len(self.deblocks) > len(self.blocks):
            x = self.deblocks[-1](x)

        return x


class AnchorHeadTemplate(nn.Module):
    def __init__(self, model_cfg, num_class, class_names, grid_size, point_cloud_range, predict_boxes_when_training, num_small_filters):
        super().__init__()
        self.model_cfg = model_cfg
        self.num_class = num_class
        self.class_names = class_names
        self.predict_boxes_when_training = predict_boxes_when_training
        self.use_multihead = self.model_cfg.get('USE_MULTIHEAD', False)

        anchor_target_cfg = self.model_cfg.TARGET_ASSIGNER_CONFIG
        self.box_coder = getattr(box_coder_utils, anchor_target_cfg.BOX_CODER)(
            num_dir_bins=anchor_target_cfg.get('NUM_DIR_BINS', 6),
            **anchor_target_cfg.get('BOX_CODER_CONFIG', {})
        )

        anchor_generator_cfg = self.model_cfg.ANCHOR_GENERATOR_CONFIG
        anchors, self.num_anchors_per_location = self.generate_anchors(
            anchor_generator_cfg, grid_size=grid_size, point_cloud_range=point_cloud_range,
            anchor_ndim=self.box_coder.code_size
        )
        self.anchors = [x.cuda() for x in anchors]
        self.target_assigner = self.get_target_assigner(anchor_target_cfg)

        self.forward_ret_dict = {}
        self.build_losses(self.model_cfg.LOSS_CONFIG)

    @staticmethod
    def generate_anchors(anchor_generator_cfg, grid_size, point_cloud_range, anchor_ndim=7):
        anchor_generator = AnchorGenerator(
            anchor_range=point_cloud_range,
            anchor_generator_config=anchor_generator_cfg
        )
        feature_map_size = [grid_size[:2] // config['feature_map_stride'] for config in anchor_generator_cfg]
        anchors_list, num_anchors_per_location_list = anchor_generator.generate_anchors(feature_map_size)

        if anchor_ndim != 7:
            for idx, anchors in enumerate(anchors_list):
                pad_zeros = anchors.new_zeros([*anchors.shape[0:-1], anchor_ndim - 7])
                new_anchors = torch.cat((anchors, pad_zeros), dim=-1)
                anchors_list[idx] = new_anchors

        return anchors_list, num_anchors_per_location_list

    def get_target_assigner(self, anchor_target_cfg):
        if anchor_target_cfg.NAME == 'ATSS':
            target_assigner = ATSSTargetAssigner(
                topk=anchor_target_cfg.TOPK,
                box_coder=self.box_coder,
                use_multihead=self.use_multihead,
                match_height=anchor_target_cfg.MATCH_HEIGHT
            )
        elif anchor_target_cfg.NAME == 'AxisAlignedTargetAssigner':
            target_assigner = AxisAlignedTargetAssigner(
                model_cfg=self.model_cfg,
                class_names=self.class_names,
                box_coder=self.box_coder,
                match_height=anchor_target_cfg.MATCH_HEIGHT
            )
        else:
            raise NotImplementedError
        return target_assigner

    def build_losses(self, losses_cfg):
        self.add_module(
            'cls_loss_func',
            loss_utils.SigmoidFocalClassificationLoss(alpha=0.25, gamma=2.0)
        )
        reg_loss_name = 'WeightedSmoothL1Loss' if losses_cfg.get('REG_LOSS_TYPE', None) is None \
            else losses_cfg.REG_LOSS_TYPE
        self.add_module(
            'reg_loss_func',
            getattr(loss_utils, reg_loss_name)(code_weights=losses_cfg.LOSS_WEIGHTS['code_weights'])
        )
        self.add_module(
            'dir_loss_func',
            loss_utils.WeightedCrossEntropyLoss()
        )

    def assign_targets(self, gt_boxes):
        """
        Args:
            gt_boxes: (B, M, 8)
        Returns:

        """
        targets_dict = self.target_assigner.assign_targets(
            self.anchors, gt_boxes
        )
        return targets_dict

    def get_cls_layer_loss(self):
        cls_preds = self.forward_ret_dict['cls_preds']
        box_cls_labels = self.forward_ret_dict['box_cls_labels']
        batch_size = int(cls_preds.shape[0])
        cared = box_cls_labels >= 0  # [N, num_anchors]
        positives = box_cls_labels > 0
        negatives = box_cls_labels == 0
        negative_cls_weights = negatives * 1.0
        cls_weights = (negative_cls_weights + 1.0 * positives).float()
        reg_weights = positives.float()
        if self.num_class == 1:
            # class agnostic
            box_cls_labels[positives] = 1

        pos_normalizer = positives.sum(1, keepdim=True).float()
        reg_weights /= torch.clamp(pos_normalizer, min=1.0)
        cls_weights /= torch.clamp(pos_normalizer, min=1.0)
        cls_targets = box_cls_labels * cared.type_as(box_cls_labels)
        cls_targets = cls_targets.unsqueeze(dim=-1)

        cls_targets = cls_targets.squeeze(dim=-1)
        one_hot_targets = torch.zeros(
            *list(cls_targets.shape), self.num_class + 1, dtype=cls_preds.dtype, device=cls_targets.device
        )
        one_hot_targets.scatter_(-1, cls_targets.unsqueeze(dim=-1).long(), 1.0)
        cls_preds = cls_preds.view(batch_size, -1, self.num_class)
        one_hot_targets = one_hot_targets[..., 1:]
        cls_loss_src = self.cls_loss_func(cls_preds, one_hot_targets, weights=cls_weights)  # [N, M]
        cls_loss = cls_loss_src.sum() / batch_size

        cls_loss = cls_loss * self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['cls_weight']
        tb_dict = {
            'rpn_loss_cls': cls_loss.item()
        }
        return cls_loss, tb_dict

    @staticmethod
    def add_sin_difference(boxes1, boxes2, dim=6):
        assert dim != -1
        rad_pred_encoding = torch.sin(boxes1[..., dim:dim + 1]) * torch.cos(boxes2[..., dim:dim + 1])
        rad_tg_encoding = torch.cos(boxes1[..., dim:dim + 1]) * torch.sin(boxes2[..., dim:dim + 1])
        boxes1 = torch.cat([boxes1[..., :dim], rad_pred_encoding, boxes1[..., dim + 1:]], dim=-1)
        boxes2 = torch.cat([boxes2[..., :dim], rad_tg_encoding, boxes2[..., dim + 1:]], dim=-1)
        return boxes1, boxes2

    @staticmethod
    def get_direction_target(anchors, reg_targets, one_hot=True, dir_offset=0, num_bins=2):
        batch_size = reg_targets.shape[0]
        anchors = anchors.view(batch_size, -1, anchors.shape[-1])
        rot_gt = reg_targets[..., 6] + anchors[..., 6]
        offset_rot = common_utils.limit_period(rot_gt - dir_offset, 0, 2 * np.pi)
        dir_cls_targets = torch.floor(offset_rot / (2 * np.pi / num_bins)).long()
        dir_cls_targets = torch.clamp(dir_cls_targets, min=0, max=num_bins - 1)

        if one_hot:
            dir_targets = torch.zeros(*list(dir_cls_targets.shape), num_bins, dtype=anchors.dtype,
                                      device=dir_cls_targets.device)
            dir_targets.scatter_(-1, dir_cls_targets.unsqueeze(dim=-1).long(), 1.0)
            dir_cls_targets = dir_targets
        return dir_cls_targets

    def get_box_reg_layer_loss(self):
        box_preds = self.forward_ret_dict['box_preds']
        box_dir_cls_preds = self.forward_ret_dict.get('dir_cls_preds', None)
        box_reg_targets = self.forward_ret_dict['box_reg_targets']
        box_cls_labels = self.forward_ret_dict['box_cls_labels']
        batch_size = int(box_preds.shape[0])

        positives = box_cls_labels > 0
        reg_weights = positives.float()
        pos_normalizer = positives.sum(1, keepdim=True).float()
        reg_weights /= torch.clamp(pos_normalizer, min=1.0)

        if isinstance(self.anchors, list):
            if self.use_multihead:
                anchors = torch.cat(
                    [anchor.permute(3, 4, 0, 1, 2, 5).contiguous().view(-1, anchor.shape[-1]) for anchor in
                     self.anchors], dim=0)
            else:
                anchors = torch.cat(self.anchors, dim=-3)
        else:
            anchors = self.anchors
        anchors = anchors.view(1, -1, anchors.shape[-1]).repeat(batch_size, 1, 1)
        box_preds = box_preds.view(batch_size, -1,
                                   box_preds.shape[-1] // self.num_anchors_per_location if not self.use_multihead else
                                   box_preds.shape[-1])
        # sin(a - b) = sinacosb-cosasinb
        box_preds_sin, reg_targets_sin = self.add_sin_difference(box_preds, box_reg_targets)
        loc_loss_src = self.reg_loss_func(box_preds_sin, reg_targets_sin, weights=reg_weights)  # [N, M]
        loc_loss = loc_loss_src.sum() / batch_size

        loc_loss = loc_loss * self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['loc_weight']
        box_loss = loc_loss
        tb_dict = {
            'rpn_loss_loc': loc_loss.item()
        }

        if box_dir_cls_preds is not None:
            dir_targets = self.get_direction_target(
                anchors, box_reg_targets,
                dir_offset=self.model_cfg.DIR_OFFSET,
                num_bins=self.model_cfg.NUM_DIR_BINS
            )

            dir_logits = box_dir_cls_preds.view(batch_size, -1, self.model_cfg.NUM_DIR_BINS)
            weights = positives.type_as(dir_logits)
            weights /= torch.clamp(weights.sum(-1, keepdim=True), min=1.0)
            dir_loss = self.dir_loss_func(dir_logits, dir_targets, weights=weights)
            dir_loss = dir_loss.sum() / batch_size
            dir_loss = dir_loss * self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['dir_weight']
            box_loss += dir_loss
            tb_dict['rpn_loss_dir'] = dir_loss.item()

        return box_loss, tb_dict

    def get_loss(self):
        cls_loss, tb_dict = self.get_cls_layer_loss()
        box_loss, tb_dict_box = self.get_box_reg_layer_loss()
        tb_dict.update(tb_dict_box)
        rpn_loss = cls_loss + box_loss

        tb_dict['rpn_loss'] = rpn_loss.item()
        return rpn_loss, tb_dict

    def generate_predicted_boxes(self, batch_size, cls_preds, box_preds, dir_cls_preds=None):
        """
        Args:
            batch_size:
            cls_preds: (N, H, W, C1)
            box_preds: (N, H, W, C2)
            dir_cls_preds: (N, H, W, C3)

        Returns:
            batch_cls_preds: (B, num_boxes, num_classes)
            batch_box_preds: (B, num_boxes, 7+C)

        """
        if isinstance(self.anchors, list):
            if self.use_multihead:
                anchors = torch.cat([anchor.permute(3, 4, 0, 1, 2, 5).contiguous().view(-1, anchor.shape[-1])
                                     for anchor in self.anchors], dim=0)
            else:
                anchors = torch.cat(self.anchors, dim=-3)
        else:
            anchors = self.anchors
        num_anchors = anchors.view(-1, anchors.shape[-1]).shape[0]
        batch_anchors = anchors.view(1, -1, anchors.shape[-1]).repeat(batch_size, 1, 1)
        batch_cls_preds = cls_preds.view(batch_size, num_anchors, -1).float() \
            if not isinstance(cls_preds, list) else cls_preds
        batch_box_preds = box_preds.view(batch_size, num_anchors, -1) if not isinstance(box_preds, list) \
            else torch.cat(box_preds, dim=1).view(batch_size, num_anchors, -1)
        batch_box_preds = self.box_coder.decode_torch(batch_box_preds, batch_anchors)

        if dir_cls_preds is not None:
            dir_offset = self.model_cfg.DIR_OFFSET
            dir_limit_offset = self.model_cfg.DIR_LIMIT_OFFSET
            dir_cls_preds = dir_cls_preds.view(batch_size, num_anchors, -1) if not isinstance(dir_cls_preds, list) \
                else torch.cat(dir_cls_preds, dim=1).view(batch_size, num_anchors, -1)
            dir_labels = torch.max(dir_cls_preds, dim=-1)[1]

            period = (2 * np.pi / self.model_cfg.NUM_DIR_BINS)
            dir_rot = common_utils.limit_period(
                batch_box_preds[..., 6] - dir_offset, dir_limit_offset, period
            )
            batch_box_preds[..., 6] = dir_rot + dir_offset + period * dir_labels.to(batch_box_preds.dtype)

        if isinstance(self.box_coder, box_coder_utils.PreviousResidualDecoder):
            batch_box_preds[..., 6] = common_utils.limit_period(
                -(batch_box_preds[..., 6] + np.pi / 2), offset=0.5, period=np.pi * 2
            )

        return batch_cls_preds, batch_box_preds

    def forward(self, **kwargs):
        raise NotImplementedError


class SingleHead(BaseBEVBackbone):
    def __init__(self, model_cfg, input_channels, num_class, num_anchors_per_location, code_size, rpn_head_cfg=None,
                 head_label_indices=None, separate_reg_config=None, num_small_filter=None):
        super().__init__(rpn_head_cfg, input_channels)

        self.num_anchors_per_location = num_anchors_per_location
        self.num_class = num_class
        self.code_size = code_size
        self.model_cfg = model_cfg
        self.separate_reg_config = separate_reg_config
        self.register_buffer('head_label_indices', head_label_indices)

        if self.num_class == 2:
            self.net = nn.Sequential(
                nn.Conv2d(num_small_filter, input_channels, 3, bias=False, padding=1),
                nn.BatchNorm2d(input_channels),
                nn.ReLU(),
                nn.Conv2d(input_channels, input_channels, 3, bias=False, padding=1),
                nn.BatchNorm2d(input_channels),
                nn.ReLU()
            )

        if self.separate_reg_config is not None:
            code_size_cnt = 0
            self.conv_box = nn.ModuleDict()
            self.conv_box_names = []
            num_middle_conv = self.separate_reg_config.NUM_MIDDLE_CONV
            num_middle_filter = self.separate_reg_config.NUM_MIDDLE_FILTER
            conv_cls_list = []
            c_in = input_channels
            for k in range(num_middle_conv):
                conv_cls_list.extend([
                    nn.Conv2d(
                        c_in, num_middle_filter,
                        kernel_size=3, stride=1, padding=1, bias=False
                    ),
                    nn.BatchNorm2d(num_middle_filter),
                    nn.ReLU()
                ])
                c_in = num_middle_filter
            conv_cls_list.append(nn.Conv2d(
                c_in, self.num_anchors_per_location * self.num_class,
                kernel_size=3, stride=1, padding=1
            ))
            self.conv_cls = nn.Sequential(*conv_cls_list)

            for reg_config in self.separate_reg_config.REG_LIST:
                reg_name, reg_channel = reg_config.split(':')
                reg_channel = int(reg_channel)
                cur_conv_list = []
                c_in = input_channels
                for k in range(num_middle_conv):
                    cur_conv_list.extend([
                        nn.Conv2d(
                            c_in, num_middle_filter,
                            kernel_size=3, stride=1, padding=1, bias=False
                        ),
                        nn.BatchNorm2d(num_middle_filter),
                        nn.ReLU()
                    ])
                    c_in = num_middle_filter

                cur_conv_list.append(nn.Conv2d(
                    c_in, self.num_anchors_per_location * int(reg_channel),
                    kernel_size=3, stride=1, padding=1, bias=True
                ))
                code_size_cnt += reg_channel
                self.conv_box[f'conv_{reg_name}'] = nn.Sequential(*cur_conv_list)
                self.conv_box_names.append(f'conv_{reg_name}')

            for m in self.conv_box.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)

            assert code_size_cnt == code_size, f'Code size does not match: {code_size_cnt}:{code_size}'
        else:
            self.conv_cls = nn.Conv2d(
                input_channels, self.num_anchors_per_location * self.num_class,
                kernel_size=1
            )
            self.conv_box = nn.Conv2d(
                input_channels, self.num_anchors_per_location * self.code_size,
                kernel_size=1
            )

        if self.model_cfg.get('USE_DIRECTION_CLASSIFIER', None) is not None:
            self.conv_dir_cls = nn.Conv2d(
                input_channels,
                self.num_anchors_per_location * self.model_cfg.NUM_DIR_BINS,
                kernel_size=1
            )
        else:
            self.conv_dir_cls = None
        self.use_multihead = self.model_cfg.get('USE_MULTIHEAD', False)
        self.init_weights()

    def init_weights(self):
        pi = 0.01
        if isinstance(self.conv_cls, nn.Conv2d):
            nn.init.constant_(self.conv_cls.bias, -np.log((1 - pi) / pi))
        else:
            nn.init.constant_(self.conv_cls[-1].bias, -np.log((1 - pi) / pi))

    def forward(self, spatial_features_2d):
        ret_dict = {}
        if self.num_class == 2:
            spatial_features_2d = self.net(spatial_features_2d)
        cls_preds = self.conv_cls(spatial_features_2d)

        if self.separate_reg_config is None:
            box_preds = self.conv_box(spatial_features_2d)
        else:
            box_preds_list = []
            for reg_name in self.conv_box_names:
                box_preds_list.append(self.conv_box[reg_name](spatial_features_2d))
            box_preds = torch.cat(box_preds_list, dim=1)

        if not self.use_multihead:
            box_preds = box_preds.permute(0, 2, 3, 1).contiguous()
            cls_preds = cls_preds.permute(0, 2, 3, 1).contiguous()
        else:
            H, W = box_preds.shape[2:]
            batch_size = box_preds.shape[0]
            box_preds = box_preds.view(-1, self.num_anchors_per_location,
                                       self.code_size, H, W).permute(0, 1, 3, 4, 2).contiguous()
            cls_preds = cls_preds.view(-1, self.num_anchors_per_location,
                                       self.num_class, H, W).permute(0, 1, 3, 4, 2).contiguous()
            box_preds = box_preds.view(batch_size, -1, self.code_size)
            cls_preds = cls_preds.view(batch_size, -1, self.num_class)

        if self.conv_dir_cls is not None:
            dir_cls_preds = self.conv_dir_cls(spatial_features_2d)
            if self.use_multihead:
                dir_cls_preds = dir_cls_preds.view(
                    -1, self.num_anchors_per_location, self.model_cfg.NUM_DIR_BINS, H, W).permute(0, 1, 3, 4,
                                                                                                  2).contiguous()
                dir_cls_preds = dir_cls_preds.view(batch_size, -1, self.model_cfg.NUM_DIR_BINS)
            else:
                dir_cls_preds = dir_cls_preds.permute(0, 2, 3, 1).contiguous()

        else:
            dir_cls_preds = None

        ret_dict['cls_preds'] = cls_preds
        ret_dict['box_preds'] = box_preds
        ret_dict['dir_cls_preds'] = dir_cls_preds

        return ret_dict


class AnchorHeadMultiDef(AnchorHeadTemplate):
    def __init__(self, model_cfg, input_channels, num_class, class_names, grid_size, point_cloud_range,
                 predict_boxes_when_training=True, num_small_filters=64):
        super().__init__(
            model_cfg=model_cfg, num_class=num_class, class_names=class_names, grid_size=grid_size,
            point_cloud_range=point_cloud_range, predict_boxes_when_training=predict_boxes_when_training,
            num_small_filters=num_small_filters
        )
        self.model_cfg = model_cfg
        self.separate_multihead = self.model_cfg.get('SEPARATE_MULTIHEAD', False)
        self.num_small_filters = num_small_filters

        if self.model_cfg.get('SHARED_CONV_NUM_FILTER', None) is not None:
            shared_conv_num_filter = self.model_cfg.SHARED_CONV_NUM_FILTER
            self.shared_conv = nn.Sequential(
                nn.Conv2d(input_channels, shared_conv_num_filter, 3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(shared_conv_num_filter, eps=1e-3, momentum=0.01),
                nn.ReLU(),
            )
        else:
            self.shared_conv = None
            shared_conv_num_filter = input_channels
        self.rpn_heads = None
        self.make_multihead(shared_conv_num_filter)

    def make_multihead(self, input_channels):
        rpn_head_cfgs = self.model_cfg.RPN_HEAD_CFGS
        rpn_heads = []
        class_names = []
        for rpn_head_cfg in rpn_head_cfgs:
            class_names.extend(rpn_head_cfg['HEAD_CLS_NAME'])

        for rpn_head_cfg in rpn_head_cfgs:
            num_anchors_per_location = sum([self.num_anchors_per_location[class_names.index(head_cls)]
                                            for head_cls in rpn_head_cfg['HEAD_CLS_NAME']])
            head_label_indices = torch.from_numpy(np.array([
                self.class_names.index(cur_name) + 1 for cur_name in rpn_head_cfg['HEAD_CLS_NAME']
            ]))

            if len(rpn_head_cfg['HEAD_CLS_NAME']) == 2:
                rpn_head = SingleHead(
                    self.model_cfg, input_channels,
                    len(rpn_head_cfg['HEAD_CLS_NAME']) if self.separate_multihead else self.num_class,
                    num_anchors_per_location, self.box_coder.code_size, rpn_head_cfg,
                    head_label_indices=head_label_indices,
                    separate_reg_config=self.model_cfg.get('SEPARATE_REG_CONFIG', None), num_small_filter=self.num_small_filters)
            else:
                rpn_head = SingleHead(
                    self.model_cfg, input_channels,
                    len(rpn_head_cfg['HEAD_CLS_NAME']) if self.separate_multihead else self.num_class,
                    num_anchors_per_location, self.box_coder.code_size, rpn_head_cfg,
                    head_label_indices=head_label_indices,
                    separate_reg_config=self.model_cfg.get('SEPARATE_REG_CONFIG', None))
            rpn_heads.append(rpn_head)
        self.rpn_heads = nn.ModuleList(rpn_heads)

    def forward(self, spatial_features_2d):
        if self.shared_conv is not None:
            spatial_features_2d = self.shared_conv(spatial_features_2d)

        ret_dicts = []
        for rpn_head in self.rpn_heads:
            if rpn_head.num_class == 2:
                ret_dicts.append(rpn_head(spatial_features_2d))
            else:
                ret_dicts.append(rpn_head(spatial_features_2d))

        cls_preds = [ret_dict['cls_preds'] for ret_dict in ret_dicts]
        box_preds = [ret_dict['box_preds'] for ret_dict in ret_dicts]
        ret = {
            'cls_preds': cls_preds if self.separate_multihead else torch.cat(cls_preds, dim=1),
            'box_preds': box_preds if self.separate_multihead else torch.cat(box_preds, dim=1),
        }

        if self.model_cfg.get('USE_DIRECTION_CLASSIFIER', False):
            dir_cls_preds = [ret_dict['dir_cls_preds'] for ret_dict in ret_dicts]
            ret['dir_cls_preds'] = dir_cls_preds if self.separate_multihead else torch.cat(dir_cls_preds, dim=1)

        self.forward_ret_dict.update(ret)

        if not self.training or self.predict_boxes_when_training:
            batch_cls_preds, batch_box_preds = self.generate_predicted_boxes(
                batch_size=1,
                cls_preds=ret['cls_preds'], box_preds=ret['box_preds'], dir_cls_preds=ret.get('dir_cls_preds', None)
            )

            if isinstance(batch_cls_preds, list):
                multihead_label_mapping = []
                for idx in range(len(batch_cls_preds)):
                    multihead_label_mapping.append(self.rpn_heads[idx].head_label_indices)

        return batch_cls_preds, batch_box_preds

    def get_cls_layer_loss(self):
        loss_weights = self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS
        if 'pos_cls_weight' in loss_weights:
            pos_cls_weight = loss_weights['pos_cls_weight']
            neg_cls_weight = loss_weights['neg_cls_weight']
        else:
            pos_cls_weight = neg_cls_weight = 1.0

        cls_preds = self.forward_ret_dict['cls_preds']
        box_cls_labels = self.forward_ret_dict['box_cls_labels']
        if not isinstance(cls_preds, list):
            cls_preds = [cls_preds]
        batch_size = int(cls_preds[0].shape[0])
        cared = box_cls_labels >= 0  # [N, num_anchors]
        positives = box_cls_labels > 0
        negatives = box_cls_labels == 0
        negative_cls_weights = negatives * 1.0 * neg_cls_weight

        cls_weights = (negative_cls_weights + pos_cls_weight * positives).float()

        reg_weights = positives.float()
        if self.num_class == 1:
            # class agnostic
            box_cls_labels[positives] = 1
        pos_normalizer = positives.sum(1, keepdim=True).float()

        reg_weights /= torch.clamp(pos_normalizer, min=1.0)
        cls_weights /= torch.clamp(pos_normalizer, min=1.0)
        cls_targets = box_cls_labels * cared.type_as(box_cls_labels)
        one_hot_targets = torch.zeros(
            *list(cls_targets.shape), self.num_class + 1, dtype=cls_preds[0].dtype, device=cls_targets.device
        )
        one_hot_targets.scatter_(-1, cls_targets.unsqueeze(dim=-1).long(), 1.0)
        one_hot_targets = one_hot_targets[..., 1:]
        start_idx = c_idx = 0
        cls_losses = 0

        for idx, cls_pred in enumerate(cls_preds):
            cur_num_class = self.rpn_heads[idx].num_class
            cls_pred = cls_pred.view(batch_size, -1, cur_num_class)
            if self.separate_multihead:
                one_hot_target = one_hot_targets[:, start_idx:start_idx + cls_pred.shape[1],
                                 c_idx:c_idx + cur_num_class]
                c_idx += cur_num_class
            else:
                one_hot_target = one_hot_targets[:, start_idx:start_idx + cls_pred.shape[1]]
            cls_weight = cls_weights[:, start_idx:start_idx + cls_pred.shape[1]]
            cls_loss_src = self.cls_loss_func(cls_pred, one_hot_target, weights=cls_weight)  # [N, M]
            cls_loss = cls_loss_src.sum() / batch_size
            cls_loss = cls_loss * loss_weights['cls_weight']
            cls_losses += cls_loss
            start_idx += cls_pred.shape[1]
        assert start_idx == one_hot_targets.shape[1]
        tb_dict = {
            'rpn_loss_cls': cls_losses.item()
        }
        return cls_losses, tb_dict

    def get_box_reg_layer_loss(self):
        box_preds = self.forward_ret_dict['box_preds']
        box_dir_cls_preds = self.forward_ret_dict.get('dir_cls_preds', None)
        box_reg_targets = self.forward_ret_dict['box_reg_targets']
        box_cls_labels = self.forward_ret_dict['box_cls_labels']

        positives = box_cls_labels > 0
        reg_weights = positives.float()
        pos_normalizer = positives.sum(1, keepdim=True).float()
        reg_weights /= torch.clamp(pos_normalizer, min=1.0)

        if not isinstance(box_preds, list):
            box_preds = [box_preds]
        batch_size = int(box_preds[0].shape[0])

        if isinstance(self.anchors, list):
            if self.use_multihead:
                anchors = torch.cat(
                    [anchor.permute(3, 4, 0, 1, 2, 5).contiguous().view(-1, anchor.shape[-1])
                     for anchor in self.anchors], dim=0
                )
            else:
                anchors = torch.cat(self.anchors, dim=-3)
        else:
            anchors = self.anchors
        anchors = anchors.view(1, -1, anchors.shape[-1]).repeat(batch_size, 1, 1)

        start_idx = 0
        box_losses = 0
        tb_dict = {}
        for idx, box_pred in enumerate(box_preds):
            box_pred = box_pred.view(
                batch_size, -1,
                box_pred.shape[-1] // self.num_anchors_per_location if not self.use_multihead else box_pred.shape[-1]
            )
            box_reg_target = box_reg_targets[:, start_idx:start_idx + box_pred.shape[1]]
            reg_weight = reg_weights[:, start_idx:start_idx + box_pred.shape[1]]
            # sin(a - b) = sinacosb-cosasinb
            if box_dir_cls_preds is not None:
                box_pred_sin, reg_target_sin = self.add_sin_difference(box_pred, box_reg_target)
                loc_loss_src = self.reg_loss_func(box_pred_sin, reg_target_sin, weights=reg_weight)  # [N, M]
            else:
                loc_loss_src = self.reg_loss_func(box_pred, box_reg_target, weights=reg_weight)  # [N, M]
            loc_loss = loc_loss_src.sum() / batch_size

            loc_loss = loc_loss * self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['loc_weight']
            box_losses += loc_loss
            tb_dict['rpn_loss_loc'] = tb_dict.get('rpn_loss_loc', 0) + loc_loss.item()

            if box_dir_cls_preds is not None:
                if not isinstance(box_dir_cls_preds, list):
                    box_dir_cls_preds = [box_dir_cls_preds]
                dir_targets = self.get_direction_target(
                    anchors, box_reg_targets,
                    dir_offset=self.model_cfg.DIR_OFFSET,
                    num_bins=self.model_cfg.NUM_DIR_BINS
                )
                box_dir_cls_pred = box_dir_cls_preds[idx]
                dir_logit = box_dir_cls_pred.view(batch_size, -1, self.model_cfg.NUM_DIR_BINS)
                weights = positives.type_as(dir_logit)
                weights /= torch.clamp(weights.sum(-1, keepdim=True), min=1.0)

                weight = weights[:, start_idx:start_idx + box_pred.shape[1]]
                dir_target = dir_targets[:, start_idx:start_idx + box_pred.shape[1]]
                dir_loss = self.dir_loss_func(dir_logit, dir_target, weights=weight)
                dir_loss = dir_loss.sum() / batch_size
                dir_loss = dir_loss * self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['dir_weight']
                box_losses += dir_loss
                tb_dict['rpn_loss_dir'] = tb_dict.get('rpn_loss_dir', 0) + dir_loss.item()
            start_idx += box_pred.shape[1]
        return box_losses, tb_dict


class pointpillar(nn.Module):

    def __init__(self, config):
        super(pointpillar, self).__init__()
        self.voxelization = voxelize(config.DATA_CONFIG.DATA_PROCESSOR[2], cfg.DATA_CONFIG.POINT_CLOUD_RANGE)
        self.vfe = PillarVFE(config.MODEL.VFE, config.DATA_CONFIG.DATA_PROCESSOR[2].VOXEL_SIZE, config.DATA_CONFIG.POINT_CLOUD_RANGE, config.DATA_CONFIG.DATA_AUGMENTOR.AUG_CONFIG_LIST[0].NUM_POINT_FEATURES)
        self.point_cloud_range = cfg.DATA_CONFIG.POINT_CLOUD_RANGE
        grid_size = (np.array(self.point_cloud_range[3:6]) - np.array(self.point_cloud_range[0:3])) / np.array(
            config.DATA_CONFIG.DATA_PROCESSOR[2].VOXEL_SIZE)
        self.grid_size = np.round(grid_size).astype(np.int64)
        self.middle_encoder = PointPillarScatter(config.MODEL.MAP_TO_BEV, self.grid_size)
        self.backbone_2d = BaseBEVBackbone(config.MODEL.BACKBONE_2D, 64)
        self.input_channels = sum(config.MODEL.BACKBONE_2D.NUM_UPSAMPLE_FILTERS)
        self.class_names = ['Car', 'Pedestrian', 'Cyclist']
        self.dense_head = AnchorHeadMultiDef(config.MODEL.DENSE_HEAD, self.input_channels, len(self.class_names), self.class_names, self.grid_size, self.point_cloud_range, num_small_filters=config.MODEL.BACKBONE_2D.NUM_FILTERS[0])

    def forward(self, points):
        x = self.voxelization(points)
        voxel_features, voxel_num_points, coords = x["voxels"], x["voxel_num_points"], x["voxel_coords"][:, 1::]
        x = self.vfe(voxel_features, voxel_num_points, coords)
        x = self.middle_encoder(x, coords)
        x = self.backbone_2d(x)
        cls_preds, box_preds = self.dense_head(x)
        return cls_preds, box_preds


if __name__ == "__main__":
    import thop
    model_weights = torch.load("/home/mk/OpenPCDet/benchmark/pointpillar_multi_define_b4/pointpillar_multi_define_b4/default/ckpt/checkpoint_epoch_80.pth")
    cfg_file = "/home/mk/OpenPCDet/tools/cfgs/kitti_models/pointpillar_multi_define_b4.yaml"
    cfg_from_yaml_file(cfg_file, cfg)
    model = pointpillar(cfg)
    model_state = model_weights["model_state"]
    del_list = []
    for key in model_state.keys():
        if key not in model.state_dict().keys():
            del_list.append(key)
    for var in del_list:
        del model_state[var]
    model.load_state_dict(model_state)
    model.eval()
    model.cuda()

    # voxelization = voxelize(cfg.DATA_CONFIG.DATA_PROCESSOR[2], cfg.DATA_CONFIG.POINT_CLOUD_RANGE)
    points_path = "/home/mk/OpenPCDet/data/kitti/testing/velodyne/000001.bin"
    inp = np.fromfile(str(points_path), dtype=np.float32).reshape(-1, 4)
    flops, params = thop.profile(model, inputs=(inp, ))
    flops, params = thop.clever_format([flops, params], "%.3f")
    print(f"flops: {flops} params: {params}")