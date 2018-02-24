import argparse

def get_args():
    parser = argparse.ArgumentParser(description='PyTorch Semantic Edge Detection Training')
    
    parser.add_argument('--checkpoint-folder', metavar='DIR',
                        help='path to checkpoint dir',
                        default='./checkpoint')
    
    parser.add_argument('--multigpu', action='store_true', 
                        help='use multiple GPUs')
    parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                        help='number of data loading workers (default: 0)')
    
    parser.add_argument('--epochs', default=150, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')

    parser.add_argument('--lr-steps', default=[50, 100], type=int, nargs="+",
                        metavar='LRSteps', help='epochs to decay learning rate by 10')
    
    parser.add_argument('-b', '--batch-size', default=64, type=int,
                        metavar='N', help='mini-batch size (default: 1)')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)')
    
    parser.add_argument('--print-freq', '-p', default=1, type=int,
                        metavar='N', help='print frequency (default: 10)')
    
    parser.add_argument('--resume-model', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--pretrained-model', default='', type=str, metavar='PATH',
                        help='path to pretrained checkpoint (default: none)')
    
    parser.add_argument('--num-classes', default=174, type=int,
                        metavar='NC', help='the number of classes for action recognition')
    
    parser.add_argument('--basemodel-name', default='resnet34', \
                        type=str, metavar='PATH',
                        help='CNN base model')
    
    args = parser.parse_args()

    return args

