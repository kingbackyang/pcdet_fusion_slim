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

    def forward_inf(self, pillar_x, pillar_y, pillar_z, pillar_i, num_voxels, x_sub_shaped, y_sub_shaped, mask):

        pillar_xyz = torch.cat((pillar_x, pillar_y, pillar_z), 2)
        points_mean = pillar_xyz.sum(dim=1, keepdim=True) / num_voxels.view(-1, 1, 1)
        f_cluster = pillar_xyz - points_mean
        f_center_offset_0 = pillar_x - x_sub_shaped
        f_center_offset_1 = pillar_y - y_sub_shaped

        features = torch.cat([pillar_xyz, pillar_i, f_cluster, f_center_offset_0, f_center_offset_1], dim=2)

        masked_features = features * mask  #
        features = self.pfn_layers[0](masked_features)
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


class BaseBEVBackboneSlim(nn.Module):
    def __init__(self, model_cfg, input_channels):
        super().__init__()
        self.model_cfg = model_cfg

        if self.model_cfg.get('LAYER_NUMS', None) is not None:
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
        # c_in_list = [input_channels, *num_filters[:-1]]
        num_filters = [input_channels] + num_filters
        self.blocks = nn.ModuleList()
        self.deblocks = nn.ModuleList()
        num_count = 0
        for idx in range(num_levels):
            cur_layers = [
                nn.ZeroPad2d(1),
                nn.Conv2d(
                    num_filters[num_count], num_filters[num_count+1], kernel_size=3,
                    stride=layer_strides[idx], padding=0, bias=False
                ),
                nn.BatchNorm2d(num_filters[num_count+1], eps=1e-3, momentum=0.01),
                nn.ReLU()
            ]
            num_count += 1
            for k in range(layer_nums[idx]):
                cur_layers.extend([
                    nn.Conv2d(num_filters[num_count], num_filters[num_count+1], kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(num_filters[num_count+1], eps=1e-3, momentum=0.01),
                    nn.ReLU()
                ])
                num_count += 1
            self.blocks.append(nn.Sequential(*cur_layers))
            if len(upsample_strides) > 0:
                stride = upsample_strides[idx]
                if stride >= 1:
                    self.deblocks.append(nn.Sequential(
                        nn.ConvTranspose2d(
                            num_filters[num_count], num_upsample_filters[idx],
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
                            num_filters[num_count-1], num_upsample_filters[idx],
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
        x = spatial_features
        for i in range(len(self.blocks)):
            x = self.blocks[i](x)
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
    def __init__(self, model_cfg, num_class, class_names, grid_size, point_cloud_range, predict_boxes_when_training):
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
        return cls_preds

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


class AnchorHeadSingle(AnchorHeadTemplate):
    def __init__(self, model_cfg, input_channels, num_class, class_names, grid_size, point_cloud_range,
                 predict_boxes_when_training=True):
        super().__init__(
            model_cfg=model_cfg, num_class=num_class, class_names=class_names, grid_size=grid_size, point_cloud_range=point_cloud_range,
            predict_boxes_when_training=predict_boxes_when_training
        )

        self.num_anchors_per_location = sum(self.num_anchors_per_location)

        self.conv_cls = nn.Conv2d(
            input_channels, self.num_anchors_per_location * self.num_class,
            kernel_size=1
        )
        self.conv_box = nn.Conv2d(
            input_channels, self.num_anchors_per_location * self.box_coder.code_size,
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
        self.init_weights()

    def init_weights(self):
        pi = 0.01
        nn.init.constant_(self.conv_cls.bias, -np.log((1 - pi) / pi))
        nn.init.normal_(self.conv_box.weight, mean=0, std=0.001)

    def forward(self, spatial_features_2d):

        cls_preds = self.conv_cls(spatial_features_2d)
        box_preds = self.conv_box(spatial_features_2d)
        dir_cls_preds = self.conv_dir_cls(spatial_features_2d)

        return cls_preds, box_preds, dir_cls_preds


class pointpillar(nn.Module):

    def __init__(self, config):
        super(pointpillar, self).__init__()
        # self.voxelization = voxelize(config.DATA_CONFIG.DATA_PROCESSOR[2], cfg.DATA_CONFIG.POINT_CLOUD_RANGE)
        self.vfe = PillarVFE(config.MODEL.VFE, config.DATA_CONFIG.DATA_PROCESSOR[2].VOXEL_SIZE, config.DATA_CONFIG.POINT_CLOUD_RANGE, config.DATA_CONFIG.DATA_AUGMENTOR.AUG_CONFIG_LIST[0].NUM_POINT_FEATURES)
        self.point_cloud_range = cfg.DATA_CONFIG.POINT_CLOUD_RANGE
        grid_size = (np.array(self.point_cloud_range[3:6]) - np.array(self.point_cloud_range[0:3])) / np.array(
            config.DATA_CONFIG.DATA_PROCESSOR[2].VOXEL_SIZE)
        self.grid_size = np.round(grid_size).astype(np.int64)
        self.middle_encoder = PointPillarScatter(config.MODEL.MAP_TO_BEV, self.grid_size)
        self.backbone_2d = BaseBEVBackboneSlim(config.MODEL.BACKBONE_2D, 64)
        self.input_channels = sum(config.MODEL.BACKBONE_2D.NUM_UPSAMPLE_FILTERS)
        self.class_names = ['Car', 'Pedestrian', 'Cyclist']
        self.dense_head = AnchorHeadSingle(config.MODEL.DENSE_HEAD, self.input_channels, len(self.class_names), self.class_names, self.grid_size, self.point_cloud_range)

    def forward(self, voxel_features, voxel_num_points, coords):
        # x = self.voxelization(points)
        # voxel_features, voxel_num_points, coords = x["voxels"], x["voxel_num_points"], x["voxel_coords"][:, 1::]
        x = self.vfe(voxel_features, voxel_num_points, coords)
        x = self.middle_encoder(x, coords)
        x = self.backbone_2d(x)
        cls_preds, box_preds = self.dense_head(x)
        return cls_preds, box_preds


class rpn(nn.Module):

    def __init__(self, config):
        super(rpn, self).__init__()
        self.point_cloud_range = cfg.DATA_CONFIG.POINT_CLOUD_RANGE
        grid_size = (np.array(self.point_cloud_range[3:6]) - np.array(self.point_cloud_range[0:3])) / np.array(
            config.DATA_CONFIG.DATA_PROCESSOR[2].VOXEL_SIZE)
        self.grid_size = np.round(grid_size).astype(np.int64)
        self.backbone_2d = BaseBEVBackboneSlim(config.MODEL.BACKBONE_2D, 64)
        self.input_channels = sum(config.MODEL.BACKBONE_2D.NUM_UPSAMPLE_FILTERS)
        self.class_names = ['Car', 'Pedestrian', 'Cyclist']
        self.dense_head = AnchorHeadSingle(config.MODEL.DENSE_HEAD, self.input_channels, len(self.class_names), self.class_names, self.grid_size, self.point_cloud_range)

    def forward(self, x):
        x = self.backbone_2d(x)
        cls_preds, box_preds, dir_cls_preds = self.dense_head(x)
        return cls_preds, box_preds, dir_cls_preds


def get_paddings_indicator(actual_num, max_num, axis=0):
    actual_num = torch.unsqueeze(actual_num, axis + 1)
    max_num_shape = [1] * len(actual_num.shape)
    max_num_shape[axis + 1] = -1
    max_num = torch.arange(max_num, dtype=torch.int, device=actual_num.device).view(max_num_shape)
    paddings_indicator = actual_num.int() > max_num
    return paddings_indicator


class pfn(nn.Module):

    def __init__(self, config):
        super(pfn, self).__init__()
        self.vfe = PillarVFE(config.MODEL.VFE, config.DATA_CONFIG.DATA_PROCESSOR[2].VOXEL_SIZE, config.DATA_CONFIG.POINT_CLOUD_RANGE, config.DATA_CONFIG.DATA_AUGMENTOR.AUG_CONFIG_LIST[0].NUM_POINT_FEATURES)

    def forward(self, pillar_x, pillar_y, pillar_z, pillar_i, num_voxels, x_sub_shaped, y_sub_shaped, z_sub_shaped, mask):

        pillar_xyz = torch.cat((pillar_x, pillar_y, pillar_z), 2)
        points_mean = pillar_xyz.sum(dim=1, keepdim=True) / num_voxels.view(-1, 1, 1)
        f_cluster = pillar_xyz - points_mean
        f_center_offset_0 = pillar_x - x_sub_shaped
        f_center_offset_1 = pillar_y - y_sub_shaped
        f_center_offset_2 = pillar_z - z_sub_shaped

        features = torch.cat([pillar_xyz, pillar_i, f_cluster, f_center_offset_0, f_center_offset_1, f_center_offset_2], dim=2)

        masked_features = features * mask  #
        features = self.vfe.pfn_layers[0](masked_features)
        features = features.squeeze()
        return features


if __name__ == "__main__":
    import json
    model_weights = torch.load("/home/mk/OpenPCDet/benchmark/slim_0.5_finetune/pointpillar/default/ckpt/checkpoint_epoch_120.pth")
    cfg_file = "/home/mk/OpenPCDet/tools/cfgs/kitti_models/pointpillar.yaml"
    cfg_from_yaml_file(cfg_file, cfg)
    slimming_json = f"/home/mk/OpenPCDet/benchmark/slim_description/slim_description_{0.5}.json"
    if len(slimming_json) != 0:
        with open(slimming_json, "r") as f:
            rpn_dict_ = json.load(f)
        channels_list = cfg["MODEL"]["BACKBONE_2D"]["NUM_UPSAMPLE_FILTERS"]
        cfg["MODEL"]["BACKBONE_2D"]["NAME"] = "BaseBEVBackboneSlim"
        cfg["MODEL"]["BACKBONE_2D"]["NUM_FILTERS"] = []
        cfg["MODEL"]["BACKBONE_2D"]["NUM_UPSAMPLE_FILTERS"] = []
        layer_num_plus = [k + 1 for k in cfg["MODEL"]["BACKBONE_2D"]["LAYER_NUMS"]]
        for i in range(len(layer_num_plus)):
            for j in range(layer_num_plus[i]):
                block_count = "blocks_{}_{}".format(i, j)
                cfg["MODEL"]["BACKBONE_2D"]["NUM_FILTERS"].append(
                    len(rpn_dict_[block_count]) if len(rpn_dict_[block_count]) != 0 else 2)
        for i in range(len(layer_num_plus)):
            deblocks_count = "deblocks_{}_0".format(i)
            cfg["MODEL"]["BACKBONE_2D"]["NUM_UPSAMPLE_FILTERS"].append(
                len(rpn_dict_[deblocks_count]) if len(rpn_dict_[deblocks_count]) != 0 else 2)
    model = pointpillar(cfg)
    model.eval()
    model.cuda()
    model_state = model_weights["model_state"]
    del_list = []
    for key in model_state.keys():
        if key not in model.state_dict().keys():
            del_list.append(key)
    for var in del_list:
        del model_state[var]

    del_part_list = []
    for key in model_state.keys():
        if not key.startswith("vfe"):
            del_part_list.append(key)
    for var in del_part_list:
        del model_state[var]

    # # model build
    voxelization = voxelize(cfg.DATA_CONFIG.DATA_PROCESSOR[2], cfg.DATA_CONFIG.POINT_CLOUD_RANGE)
    # vfe = pfn(cfg)
    #
    # # pfn infer
    # points_path = "/home/mk/OpenPCDet/data/kitti/testing/velodyne/000001.bin"
    # inp = np.fromfile(str(points_path), dtype=np.float32).reshape(-1, 4)
    # # voxelize
    # x = voxelization(inp)
    # voxels, voxel_num_points, coors = x["voxels"], x["voxel_num_points"], x["voxel_coords"]
    # vx = vfe.vfe.voxel_x
    # vy = vfe.vfe.voxel_y
    # vz = vfe.vfe.voxel_z
    # x_offset = vfe.vfe.x_offset
    # y_offset = vfe.vfe.y_offset
    # z_offset = vfe.vfe.z_offset
    # pillar_x = voxels[:, :, 0].unsqueeze(dim=-1)
    # pillar_y = voxels[:, :, 1].unsqueeze(dim=-1)
    # pillar_z = voxels[:, :, 2].unsqueeze(dim=-1)
    # pillar_i = voxels[:, :, 3].unsqueeze(dim=-1)
    # x_sub = coors[:, 3].unsqueeze(1).type_as(voxels) * vx + x_offset
    # y_sub = coors[:, 2].unsqueeze(1).type_as(voxels) * vy + y_offset
    # z_sub = coors[:, 1].unsqueeze(1).type_as(voxels) * vz + z_offset
    # ones = torch.ones([1, 32], dtype=torch.float32, device=pillar_x.device)
    # x_sub_shaped = torch.mm(x_sub, ones).unsqueeze(dim=-1)
    # y_sub_shaped = torch.mm(y_sub, ones).unsqueeze(dim=-1)
    # z_sub_shaped = torch.mm(z_sub, ones).unsqueeze(dim=-1)
    # pfn_input_mask = get_paddings_indicator(voxel_num_points, pillar_x.shape[1], axis=0)
    # pfn_input_mask = torch.unsqueeze(pfn_input_mask, -1).type_as(voxels)
    # vfe.load_state_dict(model_state)
    # vfe.eval()
    # vfe.cuda()
    # out = vfe(pillar_x, pillar_y, pillar_z, pillar_i, voxel_num_points, x_sub_shaped, y_sub_shaped, z_sub_shaped, pfn_input_mask)
    # print(out)

    # pointpillar onnx export
    pillar_x = torch.ones(12000, 32, 1).cuda()
    pillar_y = torch.ones(12000, 32, 1).cuda()
    pillar_z = torch.ones(12000, 32, 1).cuda()
    pillar_i = torch.ones(12000, 32, 1).cuda()
    x_sub_shaped = torch.ones(12000, 32, 1).cuda()
    y_sub_shaped = torch.ones(12000, 32, 1).cuda()
    z_sub_shaped = torch.ones(12000, 32, 1).cuda()
    pfn_input_num_points = torch.ones(12000).cuda()
    pfn_input_mask = torch.ones(12000, 32, 1).cuda()

    torch.onnx.export(vfe, (
    pillar_x, pillar_y, pillar_z, pillar_i, pfn_input_num_points, x_sub_shaped, y_sub_shaped, z_sub_shaped, pfn_input_mask),
                      "pfn.onnx", verbose=True, opset_version=11)

    del_rpn_list = []
    for key in model_state.keys():
        if key.startswith("vfe"):
            del_rpn_list.append(key)
    for var in del_rpn_list:
        del model_state[var]
    rpn_model = rpn(cfg)
    rpn_model.load_state_dict(model_state)
    rpn_model.eval()
    rpn_model.cuda()
    rpn_input = torch.ones(1, 64, 496, 432).cuda()
    # cls_preds, box_preds, dir_cls_preds  = rpn_model(rpn_input)
    # cls_preds = cls_preds.permute(0, 2, 3, 1).contiguous()  # [N, H, W, C]
    # box_preds = box_preds.permute(0, 2, 3, 1).contiguous()  # [N, H, W, C]
    # dir_cls_preds = dir_cls_preds.permute(0, 2, 3, 1).contiguous()
    #
    # batch_cls_preds, batch_box_preds = rpn_model.generate_predicted_boxes(
    #     batch_size=1,
    #     cls_preds=cls_preds, box_preds=box_preds, dir_cls_preds=dir_cls_preds
    # )

    torch.onnx.export(rpn_model, rpn_input,
                      "rpn.onnx", verbose=True, opset_version=10)



    # point to end
    # points_path = "/home/mk/OpenPCDet/data/kitti/testing/velodyne/000001.bin"
    # inp = np.fromfile(str(points_path), dtype=np.float32).reshape(-1, 4)
    #
    # # voxelize
    # x = voxelization(inp)
    # voxel_features, voxel_num_points, coords = x["voxels"], x["voxel_num_points"], x["voxel_coords"][:, 1::]
    # inp = (voxel_features, voxel_num_points, coords)
    #
    # cls, box = model(voxel_features, voxel_num_points, coords)
    # torch.onnx.export(model,  # model being run
    #                   inp,  # model input (or a tuple for multiple inputs)
    #                   "super_resolution.onnx",  # where to save the model (can be a file or file-like object)
    #                   verbose=True,
    #                   export_params=True,  # store the trained parameter weights inside the model file
    #                   opset_version=11,  # the ONNX version to export the model to
    #                   input_names=['voxel', 'num_points', "coords"],  # the model's input names
    #                   output_names=['cls', "box"],  # the model's output names
    #                   dynamic_axes={'voxel': {0 : "voxel_num"},
    #                                 'num_points': {0: "voxel_num"},
    #                                 "coords": {0: "voxel_num"}})
    # onnx_model = onnx.load("super_resolution.onnx")
    # onnx.checker.check_model(onnx_model)
    # print("jajjaj")