import logging
import time
import copy
import numpy as np
import torch
import math
import torch.nn.functional as F
from torch.utils.data import DataLoader, RandomSampler, SubsetRandomSampler
from torch.utils.data.distributed import DistributedSampler
import PAL
from dataset.cifar import get_cifar

from tqdm import tqdm
from utils import AverageMeter, ova_loss,\
     ova_ent, set_models, set_Wnet, test

logger = logging.getLogger(__name__)
best_acc = 0
best_acc_val = 0
best_aupr_in_ood_dic = {}
best_aupr_out_ood_dic = {}
best_roc_ood_dic = {}


def train_meta(args, meta_model, labeled_trainloader, labeled_iter, unlabeled_trainloader_all, \
            unlabeled_all_iter, coef, wnet, optimizer_wnet, meta_opt, losses_ova, losses_hat, losses_wet, output_args, default_out):
    if not args.no_progress:
        p_bar = tqdm(range(args.meta_step),
                    disable=args.local_rank not in [-1, 0])
    output_args["Meta"] = 'True'
    output_args["CLS"] = 'False'
    output_args["loss_x"] = 0
    m_s_time = time.time()
    for batch_idx in range(args.meta_step):
        try:
            feature_id, targets_x, index_x = labeled_iter.__next__()
        except:
            if args.world_size > 1:
                labeled_epoch += 1
                labeled_trainloader.sampler.set_epoch(labeled_epoch)
            labeled_iter = iter(labeled_trainloader)
            feature_id, targets_x, index_x = labeled_iter.__next__()
        try:
            feature_al, _, _ = unlabeled_all_iter.__next__()
        except:
            unlabeled_all_iter = iter(unlabeled_trainloader_all)
            feature_al, _, _ = unlabeled_all_iter.__next__()
        b_size = feature_id.shape[0]
        inputs_all = feature_al
        inputs = torch.cat([feature_id, inputs_all], 0).to(args.device)
        input_l = feature_id.to(args.device)
        targets_x = targets_x.to(args.device)
        logits, logits_open = meta_model(inputs)
        logits_open_w = logits_open[b_size:]
        weight = wnet(logits_open_w)

        norm = torch.sum(weight)
        Lx = F.cross_entropy(logits[:b_size],
                                targets_x, reduction='mean')
        Lo = ova_loss(logits_open[:b_size], targets_x)
        losses_ova.update(Lo.item())
        if batch_idx == args.eval_step - 1:
            m_Lo_e_time = time.time()
        L_o_u1, cost_w = ova_ent(logits_open_w)
        cost_w = torch.reshape(cost_w, (len(cost_w),1))
        if norm != 0:
            loss_hat =  Lx + coef * (torch.sum(weight * cost_w)/norm + Lo)

        else:
            loss_hat =  Lx + coef * (torch.sum(weight * cost_w) + Lo)

        if batch_idx == args.eval_step - 1:
            m_hat_e_time = time.time()
        
        meta_model.zero_grad()
        loss_hat.backward()
        meta_opt.step()

        losses_hat.update(loss_hat.item())
        y_l_hat, _ = meta_model(input_l)
        L_cls = F.cross_entropy(y_l_hat[:b_size],
                                targets_x, reduction='mean')
        if batch_idx == args.eval_step - 1:
            m_L_e_time = time.time()
        
        #compute upper level objective
        optimizer_wnet.zero_grad()
        L_cls.backward()
        optimizer_wnet.step()
        losses_wet.update(L_cls.item())
        output_args["loss_hat"] = losses_hat.avg
        output_args["loss_wet"] = losses_wet.avg
        if not args.no_progress:
            p_bar.set_description(default_out.format(**output_args))
            p_bar.update()
    
    Lo_time = (m_Lo_e_time - m_s_time) / 60
    hat_time = (m_hat_e_time - m_s_time) / 60
    meta_time = (m_L_e_time - m_s_time) / 60

    if not args.no_progress:
        p_bar.close()
    return meta_model, Lo_time, hat_time, meta_time


def train_cls(args, model, ema_model, labeled_trainloader, labeled_iter, optimizer, scheduler, losses, losses_x, output_args, default_out):
    if not args.no_progress:
        p_bar = tqdm(range(args.eval_step),
                    disable=args.local_rank not in [-1, 0])
    output_args["CLS"] = 'True'
    output_args["Meta"] = 'False'
    output_args["Hat"] = '0'
    output_args["Wet"] = '0'
    c_s_time = time.time()
    for batch_idx in range(args.eval_step):
        ## Data loading
        try:
            feature_id, targets_x, index_x = labeled_iter.__next__()
        except:
            if args.world_size > 1:
                labeled_epoch += 1
                labeled_trainloader.sampler.set_epoch(labeled_epoch)
            labeled_iter = iter(labeled_trainloader)
            feature_id, targets_x, index_x = labeled_iter.__next__()

        b_size = feature_id.shape[0]
        input_l = feature_id.to(args.device)
        targets_x = targets_x.to(args.device)

        ## Feed data
        logits, logits_open = model(input_l)
        ## Loss for labeled samples
        Lx = F.cross_entropy(logits[:b_size],
                                targets_x, reduction='mean')
        Lo = ova_loss(logits_open[:b_size], targets_x)
        loss = Lx + Lo
        loss.backward()

        losses.update(loss.item())
        losses_x.update(Lx.item())

        output_args["batch"] = batch_idx
        output_args["loss_x"] = losses_x.avg
        output_args["lr"] = [group["lr"] for group in optimizer.param_groups][0]

        optimizer.step()
        if args.opt != 'adam':
            scheduler.step()
        if args.use_ema:
            ema_model.update(model)
        model.zero_grad()
        if not args.no_progress:
            p_bar.set_description(default_out.format(**output_args))
            p_bar.update()
    
    c_e_time = time.time()

    c_time = (c_e_time - c_s_time)/ 60
    if not args.no_progress:
        p_bar.close()
    return model, ema_model, c_time


def train(args, labeled_trainloader, unlabeled_trainloader, test_loader, val_loader,
          ood_loaders, base_datasets, unlabeled_dataset, train_labeled_idxs, train_unlabeled_idxs, val_idxs):

    global best_acc
    global best_acc_val
    Precision = {}
    # Auroc = {}
    Recall = {}
    # end = time.time()
    labeled_trainloader_A = labeled_trainloader
    labeled_trainloader_B = labeled_trainloader
    labeled_iter_A = iter(labeled_trainloader_A)
    labeled_iter_B = iter(labeled_trainloader_B)

    default_out = "Epoch: {epoch}/{epochs:4}. " \
                  "Meta: {Meta}. " \
                  "CLS: {CLS}. " \
                  "LR: {lr:.6f}. " \
                  "Lab: {loss_x:.4f}. " \
                  "Hat: {loss_hat:.4f}. " \
                  "Wet: {loss_wet:.4f}. " 
    output_args = vars(args)

    unlabeled_dataset_all = copy.deepcopy(base_datasets)

    labeled_dataset_A = copy.deepcopy(labeled_trainloader_A.dataset)
    labeled_dataset_B = copy.deepcopy(labeled_trainloader_B.dataset)
    train_sampler = RandomSampler if args.local_rank == -1 else DistributedSampler
    labeled_trainloader_A = DataLoader(
        labeled_dataset_A,
        sampler=train_sampler(labeled_dataset_A),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        drop_last=True)
    
    labeled_trainloader_B = DataLoader(
        labeled_dataset_B,
        sampler=train_sampler(labeled_dataset_B),
        batch_size=args.batch_size,
        num_workers=args.num_workers, 
        drop_last=True)
    
    record_epoch = 0
    Meta_indexs = train_labeled_idxs
    CLS_indexs = train_labeled_idxs

    for query in tqdm(range(args.max_query)):
        print()
        print('Length of Meta model trainset:' + str(len(list(labeled_dataset_A.targets))))
        print('Length of CLS model trainset:' + str(len(list(labeled_dataset_B.targets))))
        print('-'*50)
        losses = AverageMeter()
        losses_x = AverageMeter()
        losses_hat = AverageMeter()
        losses_wet = AverageMeter()
        losses_ova = AverageMeter()

        
        if query == 0:
            class_num = args.num_classes
            ood_num = args.num_classes*2
            total_epoch = args.epochs
            # total_epoch = args.choose_epoch
        else:
            class_num = args.num_classes+1
            ood_num = (args.num_classes+1)*2
            total_epoch = args.epochs
        
        #  ready for vi compute 
        model, optimizer, scheduler = set_models(args, args.num_classes)
        wnet, optimizer_wnet = set_Wnet(args, ood_num)
        wnet.train()

        # create the meta_model to compute the vi
        meta_model, meta_opt, _ = set_models(args,class_num)
        meta_model.train()
        if args.use_ema:
            from models.ema import ModelEMA
            ema_model = ModelEMA(args, model, args.ema_decay)
        args.start_epoch = 0
        if args.local_rank != -1:
            model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank],
            output_device=args.local_rank, find_unused_parameters=True)
        
        unlabeled_trainloader_all = DataLoader(unlabeled_dataset_all,
                                                sampler=SubsetRandomSampler(train_unlabeled_idxs),
                                                batch_size=args.batch_size * args.mu,
                                                num_workers=args.num_workers,
                                                drop_last=True,
                                                )
        unlabeled_all_iter = iter(unlabeled_trainloader_all)

        model.train()
        ova_time = 0.0
        h_time = 0.0
        me_time = 0.0
        clss_time = 0.0

        for epoch in range(args.start_epoch, total_epoch):      
            output_args["epoch"] = epoch 
            coef = math.exp(-5 * (min(1 - epoch/args.epochs, 1)) ** 2)
            meta_model, Lo_time, hat_time, meta_time =train_meta(args, meta_model, labeled_trainloader_A, labeled_iter_A, unlabeled_trainloader_all, \
            unlabeled_all_iter, coef, wnet, optimizer_wnet, meta_opt, losses_ova, losses_hat, losses_wet, output_args, default_out)
            model, ema_model, c_time = train_cls(args, model, ema_model, labeled_trainloader_B, labeled_iter_B, optimizer, scheduler, losses, losses_x, output_args, default_out)

            ova_time += Lo_time
            h_time += hat_time
            me_time += meta_time
            clss_time += c_time


            if args.use_ema:
                test_model = ema_model.ema
            else:
                test_model = model
            
            if args.local_rank in [-1, 0]:
                test_loss, test_acc_close, test_overall, \
                test_unk, test_roc , roc_soft = test(args, test_loader, test_model, query)
                test_err_close = 100.0 - test_acc_close
                print("Epoch {}\t || Test Accuracy (%): {}\t || Test Error rate (%): {}\t || Test ROC rate (%): {}\t || Test ROC Soft rate (%): {}\t".format(epoch ,test_acc_close, test_err_close, test_roc, roc_soft))
                args.writer.add_scalar('train/1.train_loss', losses.avg, record_epoch)
                args.writer.add_scalar('train/2.train_loss_ova', losses_ova.avg, record_epoch)
                args.writer.add_scalar('train/2.train_loss_hat', losses_hat.avg, record_epoch)
                args.writer.add_scalar('train/2.train_loss_wnet', losses_wet.avg, record_epoch)

                args.writer.add_scalar('train/2.train_ova_time', ova_time, record_epoch)
                args.writer.add_scalar('train/2.train_hat_time', h_time, record_epoch)
                args.writer.add_scalar('train/2.train_meta_time', me_time, record_epoch)
                args.writer.add_scalar('train/2.train_class_time', clss_time, record_epoch)


                args.writer.add_scalar('test/1.test_acc', test_acc_close, record_epoch)
                args.writer.add_scalar('test/2.test_loss', test_loss, record_epoch)

                args.writer.add_scalar('test/2.test_err', test_err_close, record_epoch)
                args.writer.add_scalar('test/4.test_roc', test_roc, record_epoch)

                record_epoch += 1
        print()
 
        # AL learning
        CLS_indexs = set(CLS_indexs)
        IDIndex = []
        if args.query_strategy == 'PAL':
            IDIndex, MetaIndex, Precision[query], Recall[query] = PAL.Compute_un(args, unlabeled_trainloader_all, len(CLS_indexs), \
                meta_model, wnet, query, args.need_ID, args.miu)
        

        # 对数据集进行增减 删掉标记的ID数据
        train_unlabeled_idxs = list(set(train_unlabeled_idxs)-set(MetaIndex))
        CLS_indexs = list(CLS_indexs) + list(IDIndex)
        Meta_indexs = Meta_indexs + list(MetaIndex)
        print("Query Times: "+str(query)+" | Query Strategy: "+str(args.query_strategy)+" | Query Batch: "+str(args.query_batch)+" | Query Recall: " + str(Recall[query]) +" | Useful Query Nums: "+str(len(IDIndex))+ " | Query Precision: "+str(Precision[query]) +  " | Training Nums: "+str(len(CLS_indexs))+" | Unalebled Nums: "+str(len(train_unlabeled_idxs)))

        if args.dataset == 'cifar10':
            _, _, _, _, labeled_dataset_A, _, _, _ = get_cifar(args, list(set(train_labeled_idxs+Meta_indexs)), \
            train_unlabeled_idxs, val_idxs)
            # for B model
            _, _, _, base_datasets, labeled_dataset_B, unlabeled_dataset, test_dataset, val_dataset = get_cifar(args, CLS_indexs, \
                train_unlabeled_idxs, val_idxs)
        
        elif args.dataset == 'cifar100':
            _, _, _, _, labeled_dataset_A, _, _, _ = get_cifar(args, list(set(train_labeled_idxs+Meta_indexs)), \
            train_unlabeled_idxs, val_idxs)
            # for B model
            _, _, _, base_datasets, labeled_dataset_B, unlabeled_dataset, test_dataset, val_dataset = get_cifar(args, CLS_indexs, \
                train_unlabeled_idxs, val_idxs)
        

        labeled_dataset_A.targets[np.where(labeled_dataset_A.targets >= args.num_classes)[0]] =  args.num_classes        
        labeled_trainloader_A = DataLoader(labeled_dataset_A,
                                        sampler=train_sampler(labeled_dataset_A),
                                        batch_size=args.batch_size,
                                        num_workers=args.num_workers,
                                        drop_last=False)
        
        labeled_dataset_B.targets[np.where(labeled_dataset_B.targets >= args.num_classes)[0]] =  args.num_classes        
        labeled_trainloader_B = DataLoader(labeled_dataset_B,
                                        sampler=train_sampler(labeled_dataset_B),
                                        batch_size=args.batch_size,
                                        num_workers=args.num_workers,
                                        drop_last=False)       
        unlabeled_dataset_all = copy.deepcopy(base_datasets)

        if args.local_rank in [-1, 0]:
            args.writer.close()
