import argparse

def get_args():
    parser = argparse.ArgumentParser(description='PyTorch Semantic Edge Detection Training')
    
    parser.add_argument('--checkpoint-folder', metavar='DIR',
                        help='path to checkpoint dir',
                        default='./checkpoint')
    
    parser.add_argument('--multigpu', action='store_true', 
                        help='use multiple GPUs')
    parser.add_argument('-j', '--workers', default=16, type=int, metavar='N',
                        help='number of data loading workers (default: 0)')
    
    parser.add_argument('--epochs', default=150, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    
    parser.add_argument('--cls-num', default=20, type=int, metavar='N',
                        help='the number of classes')

    parser.add_argument('--lr-steps', default=[10000, 20000, 30000, 40000], type=int, nargs="+",
                        metavar='LRSteps', help='iterations to decay learning rate by 10')
    
    parser.add_argument('-b', '--batch-size', default=10, type=int,
                        metavar='N', help='mini-batch size (default: 1)')
    parser.add_argument('--lr', default=1e-7, type=float, metavar='L',
                        help='lr')
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
    
    args = parser.parse_args()

    return args

