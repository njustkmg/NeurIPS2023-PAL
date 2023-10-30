import argparse

__all__ = ['set_parser']

def set_parser():
    parser = argparse.ArgumentParser(description='PyTorch OpenMatch Training')
    ## Computational Configurations
    parser.add_argument('--gpu-id', default='1', type=int,
                        help='id(s) for CUDA_VISIBLE_DEVICES')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='number of workers')
    parser.add_argument('--seed', default=None, type=int,
                        help="random seed")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")
    parser.add_argument('--no-progress', default = False,
                    help="don't use progress bar")                    
    parser.add_argument('--eval_only', type=int, default=0,
                        help='1 if evaluation mode ')
    parser.add_argument('--num_classes', type=int, default=2,
                        help='for cifar10')

    parser.add_argument('--out', default='result',
                        help='directory to output the result')
    parser.add_argument('--resume', default='', type=str,
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--root', default='./data', type=str,
                        help='path to data directory')
    parser.add_argument('--dataset', default='cifar10', type=str,
                        choices=['cifar10', 'cifar100','Tiny_imagenet'],
                        help='dataset name')
    parser.add_argument('--save_dir', type=str, default='log')


    ## Hyper-parameters
    parser.add_argument('--need-ID', type=int, default=1450)
    parser.add_argument('--opt', default='sgd', type=str,
                        choices=['sgd', 'adam'],
                        help='optimize name')
    parser.add_argument('--num-labeled', type=int, default=50,
                        choices=[25, 50, 100, 400],
                        help='number of labeled data per each class')
    parser.add_argument('--num_val', type=int, default=0,
                        help='number of validation data per each class')
    parser.add_argument("--expand-labels", action="store_true",
                        help="expand labels to fit eval steps")
    parser.add_argument('--arch', default='wideresnet', type=str,
                        choices=['wideresnet', 'resnext',
                                 'resnet_imagenet'],
                        help='dataset name')
    parser.add_argument('--miu', default=1.0, type=float)
    parser.add_argument('--lambda_oem', default=0.1, type=float,
                    help='coefficient of OEM loss')
    parser.add_argument('--lambda_socr', default=0.5, type=float,
                    help='coefficient of SOCR loss, 0.5 for CIFAR10, ImageNet, '
                         '1.0 for CIFAR100')
    parser.add_argument('--start_fix', default=10, type=int,
                        help='epoch to start fixmatch training')
    parser.add_argument('--mu', default=2, type=int,
                        help='coefficient of unlabeled batch size')
    parser.add_argument('--total-steps', default=2 ** 19, type=int,
                        help='number of total steps to run')
    parser.add_argument('--epochs', default=512, type=int,
                        help='number of epochs to run')
    parser.add_argument('--threshold', default=0.0, type=float,
                        help='pseudo label threshold')
    parser.add_argument('--eval-step', default=80, type=int,
                        help='number of eval steps to run')
    parser.add_argument('--meta-step', default=100, type=int,
                        help='number of eval steps to run')
    parser.add_argument('--batch-size', default=64, type=int,
                        help='train batchsize')
    parser.add_argument('--lr', '--learning-rate', default=0.03, type=float,
                        help='initial learning rate')
    parser.add_argument('--warmup', default=0, type=float,
                        help='warmup epochs (unlabeled data based)')
    parser.add_argument('--wdecay', default=5e-4, type=float,
                        help='weight decay')
    parser.add_argument('--nesterov', action='store_true', default=True,
                        help='use nesterov momentum')
    parser.add_argument('--use-ema',  action='store_true', default=True, 
                        help='use EMA model')
    parser.add_argument('--ema-decay', default=0.999, type=float,
                        help='EMA decay rate')
    parser.add_argument('--T', default=1, type=float,
                        help='pseudo label temperature')

    # AL relate
    parser.add_argument('--max-query', type=int, default=10)
    parser.add_argument('--query-batch', type=int, default=1500)
    parser.add_argument('--query-strategy', type=str, default='random', choices=['PAL'])
    parser.add_argument('--is-filter', type=bool, default=True)
    parser.add_argument('--is-mini', type=bool, default=True)
    parser.add_argument('--lr_wnet', type=float, default=6e-5)
    
    args = parser.parse_args()
    return args
