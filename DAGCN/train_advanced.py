#!/usr/bin/python
# -*- coding:utf-8 -*-

import argparse
import os
from datetime import datetime
from utils.logger import setlogger
import logging
from utils.train_utils_combines import train_utils
import torch
import warnings
print(torch.__version__)
warnings.filterwarnings('ignore')

args = None

def parse_args():
    parser = argparse.ArgumentParser(description='Train')
    # model and data parameters
    parser.add_argument('--model_name', type=str, default='DAGCN_features', help='the name of the model')
    parser.add_argument('--data_name', type=str, default='CWRU', help='the name of the data')
    parser.add_argument('--data_dir', type=str, default=r"D:\Data\è¥¿å‚¨å¤§å­¦è½´æ‰¿æ•°æ®ä¸­å¿ƒç½‘ç«™", help='the directory of the data')
    parser.add_argument('--transfer_task', type=list, default=[[2], [1]], help='transfer learning tasks')
    parser.add_argument('--normlizetype', type=str, default='mean-std', help='nomalization type')

    # training parameters
    parser.add_argument('--cuda_device', type=str, default='0', help='assign device')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoint', help='the directory to save the model')
    parser.add_argument("--pretrained", type=bool, default=False, help='whether to load the pretrained model')
    parser.add_argument('--batch_size', type=int, default=64, help='batchsize of the training process')
    parser.add_argument('--num_workers', type=int, default=0, help='the number of training process')

    parser.add_argument('--bottleneck', type=bool, default=True, help='whether using the bottleneck layer')
    parser.add_argument('--bottleneck_num', type=int, default=256*1, help='whether using the bottleneck layer')
    parser.add_argument('--last_batch', type=bool, default=False, help='whether using the last batch')

    #
    parser.add_argument('--domain_adversarial', type=bool, default=True, help='whether use domain_adversarial')
    parser.add_argument('--hidden_size', type=int, default=1024, help='whether using the last batch')
    parser.add_argument('--trade_off_adversarial', type=str, default='Cons', help='')
    parser.add_argument('--lam_adversarial', type=float, default=1, help='this is used for Cons')

    # optimization information
    parser.add_argument('--opt', type=str, choices=['sgd', 'adam'], default='adam', help='the optimizer')
    parser.add_argument('--lr', type=float, default=1e-3, help='the initial learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='the momentum for sgd')
    parser.add_argument('--weight-decay', type=float, default=1e-5, help='the weight decay')
    parser.add_argument('--lr_scheduler', type=str, choices=['step', 'exp', 'stepLR', 'fix'], default='step', help='the learning rate schedule')
    parser.add_argument('--gamma', type=float, default=0.1, help='learning rate scheduler parameter for step and exp')
    parser.add_argument('--steps', type=str, default='150, 250', help='the learning rate decay for step and stepLR')

    # save, load and display information
    parser.add_argument('--resume', type=str, default='', help='the directory of the resume training model')
    parser.add_argument('--max_model_num', type=int, default=1, help='the number of most recent models to save')
    parser.add_argument('--middle_epoch', type=int, default=1, help='max number of epoch')
    parser.add_argument('--max_epoch', type=int, default=300, help='max number of epoch')
    parser.add_argument('--print_step', type=int, default=50, help='the interval of log training information')
    
    # æ–°å¢žï¼šä»»åŠ¡æ ‡è¯†ï¼ˆç”¨äºŽç›®å½•å‘½åï¼‰
    parser.add_argument('--task_id', type=str, default='', help='task identifier for saving results')

    args = parser.parse_args()
    return args


if __name__ == '__main__':

    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_device.strip()
    
    # Prepare the saving path for the model
    # å¦‚æžœæŒ‡å®šäº†task_idï¼Œä½¿ç”¨å¸¦ä»»åŠ¡æ ‡è¯†çš„ç›®å½•å
    if args.task_id:
        # æ ¼å¼: DAGCN_Task_3to0_20241023_164239
        timestamp = datetime.strftime(datetime.now(), '%Y%m%d_%H%M%S')
        sub_dir = f"DAGCN_{args.task_id}_{timestamp}"
    else:
        # åŽŸæ¥çš„æ ¼å¼: DAGCN_features_1023-164239
        sub_dir = args.model_name + '_' + datetime.strftime(datetime.now(), '%m%d-%H%M%S')
    
    save_dir = os.path.join(args.checkpoint_dir, sub_dir)
    
    # Ensure directory creation succeeds (including all parent directories)
    try:
        # 直接创建最终目录（递归创建所有父目录）
        os.makedirs(save_dir, exist_ok=True)
        print(f"确保目录存在: {save_dir}")
    except Exception as e:
        print(f"创建目录 {save_dir} 失败: {e}")
        raise

    # set the logger
    setlogger(os.path.join(save_dir, 'train.log'))

    # save the args
    for k, v in args.__dict__.items():
        logging.info("{}: {}".format(k, v))

    trainer = train_utils(args, save_dir)
    trainer.setup()
    trainer.train()