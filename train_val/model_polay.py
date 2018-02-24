import os
import time
import numpy as np

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.nn.functional as F

from torch.autograd import Variable

import sys
sys.path.append("../")

# Local imports
import utils.utils as utils
from utils.utils import AverageMeter


def train(args, train_loader, model, ce_criterion, optimizer, epoch, curr_lr, writer, global_step):
    batch_time = AverageMeter()
    data_time = AverageMeter()

    total_losses = AverageMeter()

    top1 = AverageMeter()
    top5 = AverageMeter()
    
    # switch to train mode
    model.train()

    end = time.time()
    for i, (img_ori_list, target, seq_len, vidID) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        # Input for Image CNN.
        img_ori_input_var = utils.check_gpu(0, img_ori_list) # BS X segments*3 X H X W
        
        bs = img_ori_list.size()[0]

        target_var = utils.check_gpu(0, target)

        # Go through the gate LSTM part to get the gate value for each frame.
        rgb_cnn_out = \
            model(img_ori_input_var) # BS X NUM_SEG X NUM_CLASSES

        loss = ce_criterion(rgb_cnn_out[:,-1,:], target_var)
        
        prec1, prec5 = utils.accuracy(rgb_cnn_out[:,-1,:].data, target_var.data, topk=(1, 5))
        top1.update(prec1[0], bs)
        top5.update(prec5[0], bs)
      
        total_losses.update(loss.data[0], bs)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if (i % args.print_freq == 0):
            print("\n\n")
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Total Loss {total_loss.val:.4f} ({total_loss.avg:.4f})\n'
                  'lr {learning_rate:.6f}\t'
                  'Top1 Acc {acc1.val:.4f} ({acc1.avg:.4f})\t'
                  'Top5 Acc {acc5.val:.4f} ({acc5.avg:.4f})\n'
                  .format(epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, total_loss=total_losses, 
                   learning_rate=curr_lr,acc1=top1, acc5=top5))
        
        global_step += 1

    return global_step

