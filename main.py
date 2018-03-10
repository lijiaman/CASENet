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

# For settings
import config

args = config.get_args()
def main():
    global args
    print("config:{0}".format(args))

    checkpoint_dir = os.path.join(args.checkpoint_folder, args.basemodel_name+ \
                    "_cnnlr_"+str(args.cnn_lr)+"_batch_"+ \
                    str(args.batch_size))

    # For visualization using TensorBoard.
    global_step = 0
    best_acc = 0

    train_loader, val_loader = prep_sth_all.get_dataloader(args)
    model = TreeCNN(args, num_segments=args.num_segments, \
                    D_feats=args.input_size, img_feats=args.img_feats, \
                    num_classes=args.num_classes, base_cnn_name=args.basemodel_name)
    if args.multigpu:
        model.cnn_net = torch.nn.DataParallel(model.cnn_net.cuda())
        model.new_fc = model.new_fc.cuda()
        model.shared_fc = model.shared_fc.cuda()
        model.top_cls = model.top_cls.cuda()
    else:
        model = model.cuda()

    optimizer = torch.optim.SGD(policies, lr=args.cnn_lr, momentum=args.momentum, weight_decay=args.weight_decay)

    cudnn.benchmark = True

    ce_criterion = nn.CrossEntropyLoss().cuda()
    
    if args.pretrained_model:
        utils.load_pretrained_model(model, args.pretrained_model)
        pretrained_acc = model_play.validate(args, val_loader, model, ce_criterion, 0, writer, global_step)

    if args.resume_model:
        checkpoint = torch.load(args.resume_model)
        args.start_epoch = checkpoint['epoch']
        best_prec1 = checkpoint['best_acc']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])

    for epoch in range(args.start_epoch, args.epochs):
        curr_lr = utils.adjust_learning_rate(args, optimizer, epoch, args.lr_steps)

        global_step = model_play.train(args, train_loader, model, ce_criterion, optimizer, epoch, curr_lr,\
                                 writer, global_step)
    
        curr_acc = model_play.validate(args, val_loader, model, ce_criterion, epoch, writer, global_step)

        if curr_acc > best_acc:
            best_acc = curr_acc
            utils.save_checkpoint({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer' : optimizer.state_dict(),
                'best_acc': best_acc,
            }, epoch, best_acc, folder=checkpoint_dir)
            print("Best Acc is {0}, in {1} epoch.".format(best_acc, epoch))

if __name__ == '__main__':
    main()

