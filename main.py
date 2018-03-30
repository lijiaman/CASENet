import os
import numpy as np

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.models as models
import torch.nn.functional as F

from torch.autograd import Variable

# Local imports
import utils.utils as utils

# For data loader
import prep_dataset.prep_SBD_dataset as prep_SBD_dataset

# For model
from modules.CASENet import CASENet_resnet101

# For training and validation
import train_val.model_play as model_play

# For visualization
import visdom
viz = visdom.Visdom(server="http://cluster7.ais.sandbox", port=22222, env='CASENet-SBD')

# For settings
import config

args = config.get_args()
def main():
    global args
    print("config:{0}".format(args))

    checkpoint_dir = args.checkpoint_folder

    global_step = 0
    min_val_loss = 999999999

    title = 'train|val loss '
    init = np.NaN
    win = viz.line(
        X=np.column_stack((np.array([init]), np.array([init]))),
        Y=np.column_stack((np.array([init]), np.array([init]))),
        opts={'title': title, 'xlabel': 'Iter', 'ylabel': 'Loss', 'legend': ['train', 'val']},
    )

    train_loader, val_loader = prep_SBD_dataset.get_dataloader(args)
    model = CASENet_resnet101(pretrained=True, num_classes=args.cls_num)

    if args.multigpu:
        model = torch.nn.DataParallel(model.cuda())
    else:
        model = model.cuda()

    policies = get_model_policy(model)
    optimizer = torch.optim.SGD(policies, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    cudnn.benchmark = True

    if args.pretrained_model:
        utils.load_pretrained_model(model, args.pretrained_model)

    if args.resume_model:
        checkpoint = torch.load(args.resume_model)
        args.start_epoch = checkpoint['epoch']+1
        min_val_loss = checkpoint['min_loss']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])

    for epoch in range(args.start_epoch, args.epochs):
        curr_lr = utils.adjust_learning_rate(args.lr, args, optimizer, global_step, args.lr_steps)

        global_step = model_play.train(args, train_loader, model, optimizer, epoch, curr_lr,\
                                 win, viz, global_step)
    
        curr_loss = model_play.validate(args, val_loader, model, epoch, win, viz, global_step)
        
        # Always store current model to avoid process crashed by accident.
        utils.save_checkpoint({
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optimizer' : optimizer.state_dict(),
            'min_loss': min_val_loss,
        }, epoch, folder=checkpoint_dir, filename="curr_checkpoint.pth.tar")

        if curr_loss < min_val_loss:
            min_val_loss = curr_loss
            utils.save_checkpoint({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer' : optimizer.state_dict(),
                'min_loss': min_val_loss,
            }, epoch, folder=checkpoint_dir)
            print("Min loss is {0}, in {1} epoch.".format(min_val_loss, epoch))

def get_model_policy(model):
    score_feats_conv_weight = []
    score_feats_conv_bias = []
    other_pts = []
    for m in model.named_modules():
        if m[0] != '' and m[0] != 'module':
            if 'score' in m[0] and isinstance(m[1], torch.nn.Conv2d):
                ps = list(m[1].parameters())
                score_feats_conv_weight.append(ps[0])
                if len(ps) == 2:
                    score_feats_conv_bias.append(ps[1])
            else:
            # elif not ('score' in m[0] and isinstance(m[1], torch.nn.Sequential)):
            # elif (not m[0].split('.')[-1].startswith('score_conv')) and (isinstance(m[1], torch.nn.Conv2d) or isinstance(m[1], ScaleLayer))
                ps = list(m[1].parameters())
                other_pts.extend(ps)
    return [
            {'params': score_feats_conv_weight, 'lr_mult': 10, 'name': 'score_conv_weight'},
            {'params': score_feats_conv_bias, 'lr_mult': 20, 'name': 'score_conv_bias'},
            {'params': filter(lambda p: p.requires_grad, other_pts), 'lr_mult': 1, 'name': 'other'},
    ]

if __name__ == '__main__':
    main()

