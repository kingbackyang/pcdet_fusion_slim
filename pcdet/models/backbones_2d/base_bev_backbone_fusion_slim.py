import numpy as np
import torch
import torch.nn as nn


class BaseBEVBackboneFusionSlim(nn.Module):
    def __init__(self, model_cfg, input_channels):
        super().__init__()
        self.model_cfg = model_cfg
        if self.model_cfg.get('FUSION_VOXEL_SPLIT', None) is not None:
            self.fusion_voxel_split = self.model_cfg.FUSION_VOXEL_SPLIT
        else:
            self.fusion_voxel_split = 1

        if self.model_cfg.get('LAYER_NUMS', None) is not None:
            layer_nums = self.model_cfg.LAYER_NUMS
            layer_strides = self.model_cfg.LAYER_STRIDES
            num_filters = self.model_cfg.NUM_FILTERS
        else:
            layer_nums = layer_strides = num_filters = []

        if self.model_cfg.get('UPSAMPLE_STRIDES', None) is not None:
            assert len(self.model_cfg.UPSAMPLE_STRIDES) == len(self.model_cfg.NUM_UPSAMPLE_FILTERS)
            self.num_upsample_filters = self.model_cfg.NUM_UPSAMPLE_FILTERS
            upsample_strides = self.model_cfg.UPSAMPLE_STRIDES
        else:
            upsample_strides = num_upsample_filters = []

        self.block2_input_filters = 0
        num_input_filters = input_channels
        fpn_fusion = 0
        self._fusion_stage = self.model_cfg.FUSION_STYLE
        if self._fusion_stage == 'late':
            self.bev_extractor = nn.Sequential(
                nn.Conv2d(self.fusion_voxel_split * 3, 8, 1, stride=layer_strides[2]),
                nn.ReLU()
            )
            self.num_upsample_filters.append(8)
        elif self._fusion_stage == 'early':
            self.bev_extractor = nn.Sequential(
                nn.Conv2d(self.fusion_voxel_split * 3, 8, 1, stride=1),
                nn.ReLU()
            )
            num_input_filters += 8
        elif self._fusion_stage == 'deep':
            self.bev_extractor = nn.Sequential(
                nn.Conv2d(self.fusion_voxel_split * 3, 8, 1, stride=layer_strides[1]),
                # BatchNorm2d(32),
                nn.ReLU(),
                # nn.MaxPool2d(2, 2),
            )
            self.block2_input_filters += 8
        elif self._fusion_stage == 'fpn_fusion':
            fpn_fusion = 1
            self.bev_extractor_block1 = nn.Sequential(
                nn.Conv2d(self.fusion_voxel_split * 3, 32, 3, padding=1, stride=layer_strides[0]),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                # nn.MaxPool2d(2, 2),
                nn.Conv2d(32, 64, 3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
            )
            self.bev_extractor_block2 = nn.Sequential(
                nn.Conv2d(64, 128, 3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.MaxPool2d(2, 2),
            )
            self.bev_extractor_block3 = nn.Sequential(
                nn.Conv2d(128, 256, 3, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(),
                nn.MaxPool2d(2, 2))
        count = 0
        self.block1 = [
            nn.ZeroPad2d(1),
            nn.Conv2d(
                num_input_filters, num_filters[count], 3, stride=layer_strides[0]),
            nn.BatchNorm2d(num_filters[count]),
            nn.ReLU()]
        count += 1
        for i in range(layer_nums[0]):
            self.block1.append(
                nn.Conv2d(num_filters[count-1], num_filters[count], 3, padding=1))
            self.block1.append(nn.BatchNorm2d(num_filters[count]))
            self.block1.append(nn.ReLU())
            count += 1
        self.block1 = nn.Sequential(*self.block1)
        self.deconv1 = nn.Sequential(
            nn.ConvTranspose2d(
                num_filters[count-1] + fpn_fusion * 64,
                self.num_upsample_filters[0],
                upsample_strides[0],
                stride=upsample_strides[0]),
            nn.BatchNorm2d(self.num_upsample_filters[0]),
            nn.ReLU(),
        )
        self.block2_input_filters += num_filters[count-1]
        self.block2 = [
            nn.ZeroPad2d(1),
            nn.Conv2d(
                self.block2_input_filters + fpn_fusion * 64,
                num_filters[count],
                3,
                stride=layer_strides[1]),
            nn.BatchNorm2d(num_filters[count]),
            nn.ReLU()]
        count += 1
        for i in range(layer_nums[1]):
            self.block2.append(
                nn.Conv2d(num_filters[count-1], num_filters[count], 3, padding=1))
            self.block2.append(nn.BatchNorm2d(num_filters[count]))
            self.block2.append(nn.ReLU())
            count += 1
        self.block2 = nn.Sequential(*self.block2)
        self.deconv2 = nn.Sequential(
            nn.ConvTranspose2d(
                num_filters[count-1] + fpn_fusion * 128,
                self.num_upsample_filters[1],
                upsample_strides[1],
                stride=upsample_strides[1]),
            nn.BatchNorm2d(self.num_upsample_filters[1]),
            nn.ReLU(),
        )
        self.block3 = [
            nn.ZeroPad2d(1),
            nn.Conv2d(num_filters[count-1] + fpn_fusion * 128, num_filters[count], 3, stride=layer_strides[2]),
            nn.BatchNorm2d(num_filters[count]),
            nn.ReLU()]
        count += 1
        for i in range(layer_nums[2]):
            self.block3.append(
                nn.Conv2d(num_filters[count-1], num_filters[count], 3, padding=1))
            self.block3.append(nn.BatchNorm2d(num_filters[count]))
            self.block3.append(nn.ReLU())
            count += 1
        self.block3 = nn.Sequential(*self.block3)
        self.deconv3 = nn.Sequential(
            nn.ConvTranspose2d(
                num_filters[count-1] + fpn_fusion * 256,
                self.num_upsample_filters[2],
                upsample_strides[2],
                stride=upsample_strides[2]),
            nn.BatchNorm2d(self.num_upsample_filters[2]),
            nn.ReLU(),
        )
        self.num_bev_features = sum(self.num_upsample_filters)

    def forward(self, data_dict):
        """
        Args:
            data_dict:
                spatial_features
        Returns:
        """
        spatial_features = data_dict['spatial_features']
        x = spatial_features
        bev = data_dict["bev_map"]

        if self._fusion_stage == 'fpn_fusion':  # fpn fusion
            x = self.block1(x)
            bev = self.bev_extractor_block1(bev)
            x = torch.cat([x, bev], dim=1)
            up1 = self.deconv1(x)

            x = self.block2(x)
            bev = self.bev_extractor_block2(bev)
            x = torch.cat([x, bev], dim=1)
            up2 = self.deconv2(x)

            x = self.block3(x)
            bev = self.bev_extractor_block3(bev)
            x = torch.cat([x, bev], dim=1)
            up3 = self.deconv3(x)
            x = torch.cat([up1, up2, up3], dim=1)

        else:
            if self._fusion_stage == 'early':  # early fusion
                x = torch.cat([x, self.bev_extractor(bev)], dim=1)
            x = self.block1(x)
            up1 = self.deconv1(x)
            if self._fusion_stage == 'deep':
                x = torch.cat([x, self.bev_extractor(bev)], dim=1)
            x = self.block2(x)
            up2 = self.deconv2(x)
            x = self.block3(x)
            up3 = self.deconv3(x)
            x = torch.cat([up1, up2, up3], dim=1)
            if self._fusion_stage == 'late':  # late fusion
                x = torch.cat([x, self.bev_extractor(bev)], dim=1)
        data_dict['spatial_features_2d'] = x

        return data_dict
