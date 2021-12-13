cd tools
CUDA_VISIBLE_DEVICES=1 python train.py --cfg_file /home/mk/OpenPCDet/tools/cfgs/kitti_models/pointpillar.yaml

CUDA_VISIBLE_DEVICES=1 python train_slim_periodic.py --cfg_file /home/mk/OpenPCDet/tools/cfgs/kitti_models/pointpillar.yaml --slim_ratio 0.15 --output benchmark/slim_0.15

CUDA_VISIBLE_DEVICES=1 python train.py --cfg_file /home/mk/OpenPCDet/tools/cfgs/kitti_models/pointpillar.yaml --output /home/mk/OpenPCDet/benchmark/slim_0.1 --slimming_json /home/mk/OpenPCDet/benchmark/slim_description/slim_description_0.1.json --slimming_pth /home/mk/OpenPCDet/benchmark/slim_description/slim_model_0.1.pth

CUDA_VISIBLE_DEVICES=1 python train_slim_periodic.py --cfg_file /home/mk/OpenPCDet/tools/cfgs/kitti_models/pointpillar.yaml --slim_ratio 0.1 --output benchmark/slim_0.1_periodic

CUDA_VISIBLE_DEVICES=1 python train_slim_periodic.py --cfg_file /home/mk/OpenPCDet/tools/cfgs/kitti_models/pointpillar.yaml --slim_ratio 0.2 --output benchmark/slim_0.2_periodic

CUDA_VISIBLE_DEVICES=0 python train_slim_periodic.py --cfg_file /home/mk/OpenPCDet/tools/cfgs/kitti_models/pointpillar.yaml --slim_ratio 0.25 --output benchmark/slim_0.25_periodic

CUDA_VISIBLE_DEVICES=1 python train_slim_periodic.py --cfg_file /home/mk/OpenPCDet/tools/cfgs/kitti_models/pointpillar.yaml --slim_ratio 0.15 --output benchmark/slim_0.15_periodic

CUDA_VISIBLE_DEVICES=1 python train.py --cfg_file /home/mk/OpenPCDet/tools/cfgs/kitti_models/pointpillar.yaml

CUDA_VISIBLE_DEVICES=1 python train.py --cfg_file /home/mk/OpenPCDet/tools/cfgs/kitti_models/pointpillar_multi_apollo.yaml --output benchmark/pointpillar_multi_apollo

CUDA_VISIBLE_DEVICES=1 python train.py --cfg_file /home/mk/OpenPCDet/tools/cfgs/kitti_models/pointpillar_multi_apollo_anchor.yaml --output benchmark/pointpillar_multi_apollo_anchor

CUDA_VISIBLE_DEVICES=0 python train.py --cfg_file /home/mk/OpenPCDet/tools/cfgs/kitti_models/pointpillar_multi_apollo_d.yaml --output benchmark/pointpillar_multi_apollo_d

CUDA_VISIBLE_DEVICES=0 python train.py --cfg_file /home/mk/OpenPCDet/tools/cfgs/kitti_models/pointpillar.yaml --output benchmark/pointpillar_origin

CUDA_VISIBLE_DEVICES=1 python train.py --cfg_file /home/mk/OpenPCDet/tools/cfgs/kitti_models/pointpillar_d.yaml --output benchmark/pointpillar_origin_d

CUDA_VISIBLE_DEVICES=1 python train.py --cfg_file /home/mk/OpenPCDet/tools/cfgs/kitti_models/pointpillar_anchor.yaml --output benchmark/pointpillar_origin_anchor

CUDA_VISIBLE_DEVICES=0 python train.py --cfg_file /home/mk/OpenPCDet/tools/cfgs/kitti_models/pointpillar_multi_define.yaml --output benchmark/pointpillar_multi_define

# four videocards

CUDA_VISIBLE_DEVICES=3 python train.py --cfg_file /home/mk/OpenPCDet/tools/cfgs/kitti_models/pointpillar_multi_apollo_b4.yaml --output benchmark/pointpillar_multi_apollo_b4

CUDA_VISIBLE_DEVICES=3 python train.py --cfg_file /home/mk/OpenPCDet/tools/cfgs/kitti_models/pointpillar_multi_apollo_anchor_b4.yaml --output benchmark/pointpillar_multi_apollo_anchor_b4

CUDA_VISIBLE_DEVICES=1 python train.py --cfg_file /home/mk/OpenPCDet/tools/cfgs/kitti_models/pointpillar_multi_apollo_d_b4.yaml --output benchmark/pointpillar_multi_apollo_d_b4

CUDA_VISIBLE_DEVICES=0 python train.py --cfg_file /home/mk/OpenPCDet/tools/cfgs/kitti_models/pointpillar_b4.yaml --output benchmark/pointpillar_origin_b4

CUDA_VISIBLE_DEVICES=1 python train.py --cfg_file /home/mk/OpenPCDet/tools/cfgs/kitti_models/pointpillar_d_b4.yaml --output benchmark/pointpillar_origin_d_b4

CUDA_VISIBLE_DEVICES=1 python train.py --cfg_file /home/mk/OpenPCDet/tools/cfgs/kitti_models/pointpillar_anchor_b4.yaml --output benchmark/pointpillar_origin_anchor_b4

CUDA_VISIBLE_DEVICES=0 python train.py --cfg_file /home/mk/OpenPCDet/tools/cfgs/kitti_models/pointpillar_multi_define_b4.yaml --output benchmark/pointpillar_multi_define_b4