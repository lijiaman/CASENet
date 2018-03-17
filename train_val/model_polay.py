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


def train(args, train_loader, model, optimizer, epoch, curr_lr, win, vis, global_step):
    batch_time = AverageMeter()
    data_time = AverageMeter()

    total_losses = AverageMeter()

    top1 = AverageMeter()
    top5 = AverageMeter()
    
    # switch to train mode
    model.train()

    end = time.time()
    for i, (img, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        
        # Input for Image CNN.
        img_var = utils.check_gpu(0, img_ori_list) # BS X segments*3 X H X W
        target_var = utils.check_gpu(0, target)
        
        bs = img.size()[0]

        score_feats5, fused_feats = model(img_var) # BS X NUM_SEG X NUM_CLASSES
       
        feats5_loss = WeightedMultiLabelSigmoidLoss(score_feats5, target_var) 
        fused_feats_loss = WeightedMultiLabelSigmoidLoss(fused_feats, target_var) 
        
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
                  .format(epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, total_loss=total_losses, 
                   learning_rate=curr_lr))
        
        global_step += 1

    return global_step

def WeightedMultiLabelSigmoidLoss(model_output, target):
    """
    model_output: BS X NUM_CLASSES X H X W
    target: BS X NUM_CLASSES X H X W 
    """
    pass
