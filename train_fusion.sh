cd tools
CUDA_VISIBLE_DEVICES=0 python train.py --cfg_file /home/mk/OpenPCDet/tools/cfgs/kitti_models/pointpillar_fusion_fpn.yaml --output /home/mk/OpenPCDet/benchmark/pp_fusion_fpn

CUDA_VISIBLE_DEVICES=1 python train.py --cfg_file /home/mk/OpenPCDet/tools/cfgs/kitti_models/pointpillar_fusion_early.yaml --output /home/mk/OpenPCDet/benchmark/pp_fusion_early

CUDA_VISIBLE_DEVICES=0 python train.py --cfg_file /home/mk/OpenPCDet/tools/cfgs/kitti_models/pointpillar_fusion_deep.yaml --output /home/mk/OpenPCDet/benchmark/pp_fusion_deep

CUDA_VISIBLE_DEVICES=1 python train.py --cfg_file /home/mk/OpenPCDet/tools/cfgs/kitti_models/pointpillar_fusion_late.yaml --output /home/mk/OpenPCDet/benchmark/pp_fusion_late

CUDA_VISIBLE_DEVICES=0 python train.py --cfg_file /home/mk/OpenPCDet/tools/cfgs/kitti_models/pointpillar_fusion_fpn_8.yaml --output /home/mk/OpenPCDet/benchmark/pp_fusion_fpn_8

CUDA_VISIBLE_DEVICES=0 python train.py --cfg_file /home/mk/OpenPCDet/tools/cfgs/kitti_models/pointpillar_fusion_early_8.yaml --output /home/mk/OpenPCDet/benchmark/pp_fusion_early_8_ndata

CUDA_VISIBLE_DEVICES=0 python train.py --cfg_file /home/mk/OpenPCDet/tools/cfgs/kitti_models/pointpillar_fusion_deep_8.yaml --output /home/mk/OpenPCDet/benchmark/pp_fusion_deep_8

CUDA_VISIBLE_DEVICES=1 python train.py --cfg_file /home/mk/OpenPCDet/tools/cfgs/kitti_models/pointpillar_fusion_late_8.yaml --output /home/mk/OpenPCDet/benchmark/pp_fusion_late_8

CUDA_VISIBLE_DEVICES=0 python train.py --cfg_file /home/mk/OpenPCDet/tools/cfgs/kitti_models/pointpillar_fusion_early_voxelsplit8_features5.yaml --output /home/mk/OpenPCDet/benchmark/pointpillar_fusion_early_voxelsplit8_features5

CUDA_VISIBLE_DEVICES=1 python train.py --cfg_file /home/mk/OpenPCDet/tools/cfgs/kitti_models/pointpillar_fusion_deep_voxelsplit8_features5.yaml --output /home/mk/OpenPCDet/benchmark/pointpillar_fusion_deep_voxelsplit8_features5

CUDA_VISIBLE_DEVICES=0 python train.py --cfg_file /home/mk/OpenPCDet/tools/cfgs/kitti_models/pointpillar_fusion_early_8.yaml --output /home/mk/OpenPCDet/benchmark/pp_fusion_early_8_ndata

