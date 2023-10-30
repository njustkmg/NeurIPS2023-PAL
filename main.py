import logging
import os
import sys
import torch
import os.path as osp
from torch.utils.tensorboard import SummaryWriter
from utils import set_model_config, \
    set_dataset, set_models, set_parser, \
    set_seed, Logger
import time

from train import train

logger = logging.getLogger(__name__)


def main():
    args = set_parser()
    global best_acc
    global best_acc_val
    sys.stdout = Logger(osp.join(args.save_dir, 'log_' + args.dataset +"_" +  str(args.miu) + '.txt'))
    print(args)
    if args.local_rank == -1:
        device = torch.device('cuda', args.gpu_id)
        args.world_size = 1
        args.n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device('cuda', args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.world_size = torch.distributed.get_world_size()
        args.n_gpu = 1
    args.device = device
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger.warning(
        f"Process rank: {args.local_rank}, "
        f"device: {args.device}, "
        f"n_gpu: {args.n_gpu}, "
        f"distributed training: {bool(args.local_rank != -1)}, ")
    logger.info(dict(args._get_kwargs()))
    if args.seed is not None:
        set_seed(args)
    if args.local_rank in [-1, 0]:
        os.makedirs(args.out, exist_ok=True)
        args.writer = SummaryWriter(args.out)
    set_model_config(args)

    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()

    if args.dataset =='cifar10':
        train_labeled_idxs, train_unlabeled_idxs, val_idxs, \
        labeled_trainloader, unlabeled_trainloader, base_datasets, unlabeled_dataset, test_loader, val_loader, ood_loaders = set_dataset(args)
    elif args.dataset =='cifar100':
        train_labeled_idxs, train_unlabeled_idxs, val_idxs, \
        labeled_trainloader, unlabeled_trainloader, base_datasets, unlabeled_dataset, test_loader, val_loader, ood_loaders = set_dataset(args)
  


  
   
    # model.zero_grad()
    if not args.eval_only:
        logger.info("***** Running training *****")
        logger.info(f"  Task = {args.dataset}@{args.num_labeled}")
        logger.info(f"  Num Epochs = {args.epochs}")
        logger.info(f"  Batch size per GPU = {args.batch_size}")
        logger.info(f"  Total train batch size = {args.batch_size*args.world_size}")
        logger.info(f"  Total optimization steps = {args.total_steps}")
        start_time = time.time()
        train(args, labeled_trainloader, unlabeled_trainloader, test_loader, val_loader,
          ood_loaders, base_datasets, unlabeled_dataset, train_labeled_idxs, train_unlabeled_idxs, val_idxs)
        end_time = time.time()
        running_time_seconds = end_time - start_time
        # Convert running time to hours
        running_time_hours = running_time_seconds / 3600
        # Print the result
        print(f"Algorithm running time: {running_time_hours:.3f} hours")
   
    # else:
        # logger.info("***** Running Evaluation *****")
        # logger.info(f"  Task = {args.dataset} @{args.num_labeled}")
        # eval_model(args, labeled_trainloader, unlabeled_dataset, test_loader, val_loader,
            #  ood_loaders, model, ema_model)


if __name__ == '__main__':
    main()

