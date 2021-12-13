cd tools
CUDA_VISIBLE_DEVICES=0 python train.py --cfg_file /media/zjurobot/403/yangjingru/pcdet/tools/cfgs/kitti_models/pointpillar.yaml --slimming_json /media/zjurobot/403/yangjingru/pcdet/benchmark/slim_description/slim_description_0.1.json --slimming_pth /media/zjurobot/403/yangjingru/pcdet/benchmark/slim_description/slim_model_0.1.pth --output benchmark/slim_0.1_finetune

CUDA_VISIBLE_DEVICES=1 python train.py --cfg_file /media/zjurobot/403/yangjingru/pcdet/tools/cfgs/kitti_models/pointpillar.yaml --slimming_json /media/zjurobot/403/yangjingru/pcdet/benchmark/slim_description/slim_description_0.2.json --slimming_pth /media/zjurobot/403/yangjingru/pcdet/benchmark/slim_description/slim_model_0.2.pth --output benchmark/slim_0.2_finetune

CUDA_VISIBLE_DEVICES=2 python train.py --cfg_file /media/zjurobot/403/yangjingru/pcdet/tools/cfgs/kitti_models/pointpillar.yaml --slimming_json /media/zjurobot/403/yangjingru/pcdet/benchmark/slim_description/slim_description_0.3.json --slimming_pth /media/zjurobot/403/yangjingru/pcdet/benchmark/slim_description/slim_model_0.3.pth --output benchmark/slim_0.3_finetune

CUDA_VISIBLE_DEVICES=3 python train.py --cfg_file /media/zjurobot/403/yangjingru/pcdet/tools/cfgs/kitti_models/pointpillar.yaml --slimming_json /media/zjurobot/403/yangjingru/pcdet/benchmark/slim_description/slim_description_0.4.json --slimming_pth /media/zjurobot/403/yangjingru/pcdet/benchmark/slim_description/slim_model_0.4.pth --output benchmark/slim_0.4_finetune

CUDA_VISIBLE_DEVICES=0 python train.py --cfg_file /home/mk/OpenPCDet/tools/cfgs/kitti_models/pointpillar.yaml --slimming_json /home/mk/OpenPCDet/benchmark/slim_description/slim_description_0.5.json --slimming_pth /home/mk/OpenPCDet/benchmark/slim_description/slim_model_0.5.pth --output benchmark/slim_0.5_finetune

CUDA_VISIBLE_DEVICES=1 python train.py --cfg_file /home/mk/OpenPCDet/tools/cfgs/kitti_models/pointpillar.yaml --slimming_json /home/mk/OpenPCDet/benchmark/slim_description/slim_description_0.6.json --slimming_pth /home/mk/OpenPCDet/benchmark/slim_description/slim_model_0.6.pth --output benchmark/slim_0.6_finetune

CUDA_VISIBLE_DEVICES=0 python train.py --cfg_file /home/mk/OpenPCDet/tools/cfgs/kitti_models/pointpillar.yaml --slimming_json /home/mk/OpenPCDet/benchmark/slim_description/slim_description_0.15.json --slimming_pth /home/mk/OpenPCDet/benchmark/slim_description/slim_model_0.15.pth --output benchmark/slim_0.15_finetune

CUDA_VISIBLE_DEVICES=1 python train.py --cfg_file /home/mk/OpenPCDet/tools/cfgs/kitti_models/pointpillar.yaml --slimming_json /home/mk/OpenPCDet/benchmark/slim_description/slim_description_0.25.json --slimming_pth /home/mk/OpenPCDet/benchmark/slim_description/slim_model_0.25.pth --output benchmark/slim_0.25_finetune

CUDA_VISIBLE_DEVICES=2 python train.py --cfg_file /home/mk/OpenPCDet/tools/cfgs/kitti_models/pointpillar.yaml --slimming_json /home/mk/OpenPCDet/benchmark/slim_description/slim_description_0.35.json --slimming_pth /home/mk/OpenPCDet/benchmark/slim_description/slim_model_0.35.pth --output benchmark/slim_0.35_finetune

CUDA_VISIBLE_DEVICES=3 python train.py --cfg_file /home/mk/OpenPCDet/tools/cfgs/kitti_models/pointpillar.yaml --slimming_json /home/mk/OpenPCDet/benchmark/slim_description/slim_description_0.45.json --slimming_pth /home/mk/OpenPCDet/benchmark/slim_description/slim_model_0.45.pth --output benchmark/slim_0.45_finetune

CUDA_VISIBLE_DEVICES=0 python train.py --cfg_file /home/mk/OpenPCDet/tools/cfgs/kitti_models/pointpillar.yaml --slimming_json /home/mk/OpenPCDet/benchmark/slim_description/slim_description_0.55.json --slimming_pth /home/mk/OpenPCDet/benchmark/slim_description/slim_model_0.55.pth --output benchmark/slim_0.55_finetune

