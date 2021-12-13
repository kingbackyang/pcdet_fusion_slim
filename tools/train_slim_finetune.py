import argparse
import datetime
import glob
import os
from pathlib import Path
from test import repeat_eval_ckpt
import json

import torch
import torch.distributed as dist
import torch.nn as nn
from tensorboardX import SummaryWriter

from pcdet.config import cfg, cfg_from_list, cfg_from_yaml_file, log_config_to_file
from pcdet.datasets import build_dataloader
from pcdet.models import build_network, model_fn_decorator
from pcdet.utils import common_utils
from train_utils.optimization import build_optimizer, build_scheduler
from train_utils.train_utils import train_model


def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default=None, help='specify the config for training')

    parser.add_argument('--batch_size', type=int, default=None, required=False, help='batch size for training')
    parser.add_argument('--epochs', type=int, default=None, required=False, help='number of epochs to train for')
    parser.add_argument('--workers', type=int, default=8, help='number of workers for dataloader')
    parser.add_argument('--extra_tag', type=str, default='default', help='extra tag for this experiment')
    parser.add_argument('--ckpt', type=str, default=None, help='checkpoint to start from')
    parser.add_argument('--pretrained_model', type=str, default=None, help='pretrained_model')
    parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm'], default='none')
    parser.add_argument('--tcp_port', type=int, default=18888, help='tcp port for distrbuted training')
    parser.add_argument('--sync_bn', action='store_true', default=False, help='whether to use sync bn')
    parser.add_argument('--fix_random_seed', action='store_true', default=False, help='')
    parser.add_argument('--ckpt_save_interval', type=int, default=1, help='number of training epochs')
    parser.add_argument('--local_rank', type=int, default=0, help='local rank for distributed training')
    parser.add_argument('--max_ckpt_save_num', type=int, default=30, help='max number of saved checkpoint')
    parser.add_argument('--merge_all_iters_to_one_epoch', action='store_true', default=False, help='')
    parser.add_argument('--set', dest='set_cfgs', default=None, nargs=argparse.REMAINDER,
                        help='set extra config keys if needed')

    parser.add_argument('--max_waiting_mins', type=int, default=0, help='max waiting minutes')
    parser.add_argument('--start_epoch', type=int, default=0, help='')
    parser.add_argument('--save_to_file', action='store_true', default=False, help='')
    parser.add_argument('--output', default='', help='')
    parser.add_argument('--slimming_json', default="", help='the dir to save logs and models')
    parser.add_argument('--slim_ratio', default=0.15, help='the dir to save logs and models')

    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)
    cfg.TAG = Path(args.cfg_file).stem
    cfg.EXP_GROUP_PATH = '/'.join(args.cfg_file.split('/')[1:-1])  # remove 'cfgs' and 'xxxx.yaml'

    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs, cfg)

    return args, cfg


def main():
    args, cfg = parse_config()
    if args.launcher == 'none':
        dist_train = False
        total_gpus = 1
    else:
        total_gpus, cfg.LOCAL_RANK = getattr(common_utils, 'init_dist_%s' % args.launcher)(
            args.tcp_port, args.local_rank, backend='nccl'
        )
        dist_train = True
    if args.batch_size is None:
        args.batch_size = cfg.OPTIMIZATION.BATCH_SIZE_PER_GPU
    else:
        assert args.batch_size % total_gpus == 0, 'Batch size should match the number of gpus'
        args.batch_size = args.batch_size // total_gpus

    args.epochs = cfg.OPTIMIZATION.NUM_EPOCHS if args.epochs is None else args.epochs

    if args.fix_random_seed:
        common_utils.set_random_seed(666)
    log_file = 'log_train.txt'
    logger = common_utils.create_logger(log_file, rank=cfg.LOCAL_RANK)
    train_set, train_loader, train_sampler = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        batch_size=args.batch_size,
        dist=dist_train, workers=args.workers,
        logger=logger,
        training=True,
        merge_all_iters_to_one_epoch=args.merge_all_iters_to_one_epoch,
        total_epochs=args.epochs
    )
    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=train_set)
    if args.sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    model.cuda()
    network_weights = torch.load(args.ckpt)
    model.load_state_dict(network_weights["model_state"])
    total = 0
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            total += m.weight.data.shape[0]

    bn = torch.zeros(total)
    index = 0
    rpn_key_list = []
    depth_dict = {}
    rpn_list = []
    for i in range(3):
        count = 0
        for m in model.backbone_2d.blocks[i]:
            if isinstance(m, nn.BatchNorm2d):
                size = m.weight.data.shape[0]
                rpn_list.append(size)
                bn[index:(index + size)] = m.weight.data.abs().clone()
                key = "blocks_{}_{}".format(i, count)
                rpn_key_list.append(key)
                depth_dict[key] = list(range(index, index + size))
                index += size
                count += 1
    for i in range(3):
        count = 0
        for m in model.backbone_2d.deblocks[i]:
            if isinstance(m, nn.BatchNorm2d):
                size = m.weight.data.shape[0]
                rpn_list.append(size)
                key = "deblocks_{}_{}".format(i, count)
                rpn_key_list.append(key)
                depth_dict[key] = list(range(index, index + size))
                bn[index:(index + size)] = m.weight.data.abs().clone()
                count += 1
                index += size
    print("index: ", index)
    y, ind = torch.sort(bn)
    thre_index = int(index * args.slim_ratio)
    thre = y[thre_index]
    ind_pick = torch.where(y > thre)
    ind_end = ind[ind_pick]
    ind_end = [int(v) for v in list(ind_end.numpy())]
    for key in depth_dict.keys():
        value_list = depth_dict[key]
        tmp = []
        for var in ind_end:
            if var in value_list:
                tmp.append(var)
        tmp.sort()
        depth_dict[key] = tmp
    with open(f"/home/mk/OpenPCDet/benchmark/slim_description/slim_description_{args.slim_ratio}.json", "w") as f:
        json.dump(depth_dict, f)

    rpn_dict_ = depth_dict
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
        cfg["MODEL"]["BACKBONE_2D"]["NUM_UPSAMPLE_FILTERS"].append(len(rpn_dict_[deblocks_count]) if len(rpn_dict_[deblocks_count]) != 0 else 2)
    model_slim = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=train_set)
    signal_num_plus = [k + 1 for k in cfg["MODEL"]["BACKBONE_2D"]["LAYER_NUMS"]]
    signal_list = [int(sum(signal_num_plus[:int(i + 1)]) - 1) for i in range(len(signal_num_plus))]
    signal = 0
    deconv_cfg_list = []
    rpn_dict = depth_dict
    for i in range(3):
        for m in model.backbone_2d.blocks[i]:
            if isinstance(m, nn.Conv2d):
                if signal == 0:
                    key = rpn_key_list[signal]
                    mask_cfg = rpn_dict[key]
                    if signal == 0:
                        mask_total = 0
                    else:
                        mask_total = sum(rpn_list[:int(signal)])
                    if len(mask_cfg) == 0:
                        mask_cfg.append(mask_total)
                        mask_cfg.append(int(mask_total + 1))
                        rpn_dict[key] = mask_cfg
                    mask_cfg = torch.tensor(mask_cfg) - mask_total
                    rpn_conv_length = rpn_list[signal]

                    mask = torch.zeros(rpn_conv_length)
                    mask_cfg = torch.tensor(mask_cfg)
                    mask[mask_cfg] = 1
                    m.weight.data = m.weight.data[mask_cfg, :, :, :]
                    print("signal: {} conv shape: {}".format(signal, m.weight.data.shape))
                    if m.bias:
                        m.bias.data = m.bias.data[mask_cfg, :, :, :]
                        # m.bias.data = m.bias.data[:, mask_cfg_after, :, :]
                    signal = signal + 1
                else:
                    m.weight.data = m.weight.data[:, mask_cfg, :, :]
                    # after
                    key = rpn_key_list[signal]
                    mask_cfg = rpn_dict[key]
                    if signal == 0:
                        mask_total = 0
                    else:
                        mask_total = sum(rpn_list[:int(signal)])
                    if len(mask_cfg) == 0:
                        mask_cfg.append(mask_total)
                        mask_cfg.append(int(mask_total + 1))
                        rpn_dict[key] = mask_cfg
                    mask_cfg = torch.tensor(mask_cfg) - mask_total
                    rpn_conv_length = rpn_list[signal]
                    if signal in signal_list:
                        deconv_cfg_list.append(mask_cfg)
                    mask = torch.zeros(rpn_conv_length)
                    mask_cfg = torch.tensor(mask_cfg)
                    mask[mask_cfg] = 1

                    m.weight.data = m.weight.data[mask_cfg, :, :, :]
                    print("signal: {} conv shape: {}".format(signal, m.weight.data.shape))
                    if m.bias:
                        m.bias.data = m.bias.data[mask_cfg, :, :, :]
                        # m.bias.data = m.bias.data[:, mask_cfg_after, :, :]
                    signal = signal + 1

            if isinstance(m, nn.BatchNorm2d):
                mask_cfg_ = list(mask_cfg.numpy())
                m.weight.data = m.weight.data[mask_cfg_]
                m.bias.data = m.bias.data[mask_cfg_]
                m.running_mean = m.running_mean[mask_cfg_]
                m.running_var = m.running_var[mask_cfg_]
                print("signal: {} bn shape: {}".format(signal, m.weight.data.shape[0]))
    neck_list = []
    for i in range(3):
        for m in model.backbone_2d.deblocks[i]:
            if isinstance(m, nn.ConvTranspose2d):
                m.weight.data = m.weight.data[deconv_cfg_list[i], :, :, :]
                key = rpn_key_list[signal]
                mask_cfg = rpn_dict[key]
                if signal == 0:
                    mask_total = 0
                else:
                    mask_total = sum(rpn_list[:int(signal)])
                if len(mask_cfg) == 0:
                    mask_cfg.append(mask_total)
                    mask_cfg.append(int(mask_total + 1))
                    rpn_dict[key] = mask_cfg
                mask_cfg = torch.tensor(mask_cfg) - mask_total
                neck_list.append(mask_cfg)
                rpn_conv_length = rpn_list[signal]

                mask = torch.zeros(rpn_conv_length)
                mask_cfg = torch.tensor(mask_cfg)
                mask[mask_cfg] = 1
                m.weight.data = m.weight.data[:, mask_cfg, :, :]
                print("signal: {} conv shape: {}".format(signal, m.weight.data.shape))
                if m.bias:
                    m.bias.data = m.bias.data[:, mask_cfg, :, :]
                signal = signal + 1

            if isinstance(m, nn.BatchNorm2d):
                mask_cfg = list(mask_cfg.numpy())
                m.weight.data = m.weight.data[mask_cfg]
                m.bias.data = m.bias.data[mask_cfg]
                m.running_mean = m.running_mean[mask_cfg]
                m.running_var = m.running_var[mask_cfg]
                print("signal: {} bn shape: {}".format(signal, m.weight.data.shape[0]))
    channels_mile = [int(sum(channels_list[:int(i + 1)])) for i in range(len(channels_list))]
    for m in model.dense_head.modules():
        tensor_list = []
        if isinstance(m, nn.Conv2d):
            for i in range(len(channels_mile)):
                if i == 0:
                    part = m.weight.data[:, 0:channels_mile[i], :, :]
                else:
                    part = m.weight.data[:, channels_mile[int(i - 1)]:channels_mile[i], :, :]
                tensor_list.append(part[:, neck_list[i], :, :])
            tensor_concat = torch.cat(tensor_list, dim=1)
            m.weight.data = tensor_concat
            print(m.weight.data.shape)
    model_slim.load_state_dict(model.state_dict())
    torch.save(model_slim.state_dict(), f"/home/mk/OpenPCDet/benchmark/slim_description/slim_model_{args.slim_ratio}.pth")


if __name__ == '__main__':
    main()
