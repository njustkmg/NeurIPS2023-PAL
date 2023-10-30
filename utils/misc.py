'''Some helper functions for PyTorch, including:
    - get_mean_and_std: calculate the mean and std value of dataset.
'''
import logging
import time
import pandas as pd
from tqdm import tqdm
import torch.nn.functional as F
import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, ConcatDataset, Subset
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from dataset.cifar import Use_Subset
import seaborn as sns

logger = logging.getLogger(__name__)

__all__ = ['get_mean_and_std', 'accuracy', 'AverageMeter',
           'accuracy_open', 'ova_loss', 'compute_roc',
           'compute_roc_aupr', 'misc_id_ood', 'ova_ent', 'exclude_dataset',
           'test_ood', 'test','test_feat', 'multiclass_auroc','compute_S_ID']






def get_mean_and_std(dataset):
    '''Compute the mean and std value of dataset.'''
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=4)

    mean = torch.zeros(3)
    std = torch.zeros(3)
    logger.info('==> Computing mean and std..')
    for inputs, targets in dataloader:
        for i in range(3):
            mean[i] += inputs[:, i, :, :].mean()
            std[i] += inputs[:, i, :, :].std()
    mean.div_(len(dataset))
    std.div_(len(dataset))
    return mean, std


def accuracy(output, target, topk=(1,5)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))

    res = []

    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def accuracy_open(pred, target, topk=(1,5), num_classes=5):
    """Computes the precision@k for the specified values of k,
    num_classes are the number of known classes.
    This function returns overall accuracy,
    accuracy to reject unknown samples,
    the size of unknown samples in this batch."""
    maxk = max(topk)
    batch_size = target.size(0)
    pred = pred.view(-1, 1)
    pred = pred.t()
    ind = (target == num_classes)
    unknown_size = len(ind)
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    if ind.sum() > 0:
        unk_corr = pred.eq(target).view(-1)[ind]
        acc = torch.sum(unk_corr).item() / unk_corr.size(0)
    else:
        acc = 0

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res[0], acc, unknown_size


def compute_roc(unk_all, label_all, num_known):
    Y_test = np.zeros(unk_all.shape[0])
    # 将ood的数据部分设置为1，id数据部分设置为0
    unk_pos = np.where(label_all >= num_known)[0]
    Y_test[unk_pos] = 1
    return roc_auc_score(Y_test, unk_all)


def multiclass_auroc(y_true, y_pred, target_label):
    y_true = np.array(y_true)
    y_true_class = np.where(y_true == target_label, 1, 0)
    y_pred_class = np.array(y_pred)[:, target_label]
    roc = roc_auc_score(y_true_class, y_pred_class)
    return roc


def compute_roc_aupr(unk_all, label_all, num_known):
    '''
    roc: 以ood数据为positive
    aupr_out: 以ood为positive
    aupr_in: 以in为positive
    '''
    Y_test = np.zeros(unk_all.shape[0])
    unk_pos = np.where(label_all >= num_known)[0]
    Y_test[unk_pos] = 1
    return roc_auc_score(Y_test, unk_all), \
           average_precision_score(Y_test, unk_all), \
           average_precision_score(1 - Y_test, -1.0 * unk_all)


def misc_id_ood(score_id, score_ood):
    '''
    roc: 以in数据为positive
    aupr_in：以in为positive
    aupr_out：以ood为positive
    '''
    id_all = np.r_[score_id, score_ood]
    Y_test = np.zeros(score_id.shape[0]+score_ood.shape[0])
    Y_test[score_id.shape[0]:] = 1
    return roc_auc_score(Y_test, id_all),\
           average_precision_score(Y_test, id_all), \
           average_precision_score(1 - Y_test, -1.0 * id_all)


def ova_loss(logits_open, label):
    logits_open = logits_open.view(logits_open.size(0), 2, -1)
    logits_open = F.softmax(logits_open, 1)
    label_s_sp = torch.zeros((logits_open.size(0),
                              logits_open.size(2))).long().to(label.device)
    label_range = torch.arange(0, logits_open.size(0)).long()
    label_s_sp[label_range, label] = 1
    label_sp_neg = 1 - label_s_sp
    open_loss = torch.mean(torch.sum(-torch.log(logits_open[:, 1, :]
                                                + 1e-8) * label_s_sp, 1))
    open_loss_neg = torch.mean(torch.max(-torch.log(logits_open[:, 0, :]
                                                    + 1e-8) * label_sp_neg, 1)[0])
    open_l = torch.sum(-torch.log(logits_open[:, 1, :]
                                                + 1e-8) * label_s_sp, 1)
    open_l_neg = torch.max(-torch.log(logits_open[:, 0, :]
                                                    + 1e-8) * label_sp_neg, 1)[0]
    # print(open_l.shape)
    # print(open_l_neg.shape)
    Lo = open_loss_neg + open_loss
    return Lo


def compute_S_ID(logits_open):
    logits_open = logits_open.view(logits_open.size(0), 2, -1)
    logits_open = F.softmax(logits_open, 1)
    logits_open, _ = torch.max(logits_open, dim=2)
    logits_open  = logits_open[:,0]
    L_c = 2.5*(1+logits_open *torch.log(logits_open + 1e-8))
    return L_c


def ova_ent(logits_open):
    logits_open = logits_open.view(logits_open.size(0), 2, -1)
    logits_open = F.softmax(logits_open, 1)
    Le = torch.mean(torch.mean(torch.sum(-logits_open *
                                   torch.log(logits_open + 1e-8), 1), 1))
    L_c = torch.mean(torch.sum(-logits_open *
                                   torch.log(logits_open + 1e-8), 1), 1)
    return Le, L_c


class AverageMeter(object):
    """Computes and stores the average and current value
       Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def exclude_dataset(args, dataset, model, exclude_known=False):
    data_time = AverageMeter()
    end = time.time()
    dataset.init_index()
    test_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        drop_last=False,
        shuffle=False)
    if not args.no_progress:
        test_loader = tqdm(test_loader,
                           disable=args.local_rank not in [-1, 0])
    model.eval()
    with torch.no_grad(): 
        for batch_idx, ((_, _, inputs), targets, index) in enumerate(test_loader):
            data_time.update(time.time() - end)
            inputs = inputs.to(args.device)
            outputs, outputs_open = model(inputs)
            outputs = F.softmax(outputs, 1)
            out_open = F.softmax(outputs_open.view(outputs_open.size(0), 2, -1), 1)
            tmp_range = torch.arange(0, out_open.size(0)).long().cuda(args.device)
            pred_close = outputs.data.max(1)[1]
            unk_score = out_open[tmp_range, 0, pred_close]
            known_ind = unk_score < 0.5
            if batch_idx == 0:
                known_all = known_ind
            else:
                known_all = torch.cat([known_all, known_ind], 0)
        if not args.no_progress:
            test_loader.close()
    known_all = known_all.data.cpu().numpy()
    if exclude_known:
        ind_selected = np.where(known_all == 0)[0]
    else:
        ind_selected = np.where(known_all != 0)[0]
    print("selected ratio %s"%( (len(ind_selected)/ len(known_all))))
    model.train()
    dataset.set_index(ind_selected)


def test_feat(args, test_loader, model):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    end = time.time()
    if not args.no_progress:
        test_loader = tqdm(test_loader,
                           disable=args.local_rank not in [-1, 0])
    flag = True
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            data_time.update(time.time() - end)
            model.eval()
            inputs = inputs.to(args.device)
            targets = targets.to(args.device)
            feat = model(inputs, feat_only = True)
            if flag:
                flag = False
                outputs, outputs_open = model(inputs)
                args.num_classes = int(outputs.size(1))
                print('feat size: ',feat.size())
                print('targets size: ',targets.size())
                print('num_classes: ',args.num_classes)
            targets_unk = targets >= args.num_classes
            targets[targets_unk] = args.num_classes
            if batch_idx == 0:
                feat_all = feat
                label_all = targets
            else:
                feat_all = torch.cat([feat_all, feat], 0)
                label_all = torch.cat([label_all, targets], 0)
            
            batch_time.update(time.time() - end)
            end = time.time()
    feat_all = feat_all.data.cpu().numpy()
    label_all = label_all.data.cpu().numpy()
    return feat_all, label_all


def test(args, test_loader, model, query=-1, val=False, feature=False):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    acc = AverageMeter()
    unk = AverageMeter()
    top5 = AverageMeter()
    end = time.time()

    if not args.no_progress:
        test_loader = tqdm(test_loader,
                           disable=args.local_rank not in [-1, 0])
    with torch.no_grad():
        for batch_idx, data in enumerate(test_loader):
            if len(data) == 2:
                inputs, targets = data 
            else:
                inputs, targets, index = data
            data_time.update(time.time() - end)
            model.eval()
            inputs = inputs.to(args.device)
            targets = targets.to(args.device)
            outputs, outputs_open = model(inputs)

            if feature:
                feat = model(inputs, feat_only = True)
                if batch_idx == 0:
                    feat_all = feat
                else:
                    feat_all = torch.cat([feat_all, feat], 0)
            outputs = F.softmax(outputs, 1)
            out_open = F.softmax(outputs_open.view(outputs_open.size(0), 2, -1), 1)
            tmp_range = torch.arange(0, out_open.size(0)).long().cuda(args.device)
            pred_close = outputs.data.max(1)[1]
            unk_score = out_open[tmp_range, 0, pred_close]
            known_score = outputs.max(1)[0]
            targets_unk = targets >= int(outputs.size(1))
            targets[targets_unk] = int(outputs.size(1))
            known_targets = targets < int(outputs.size(1))#[0]
            known_pred = outputs[known_targets]
            known_targets = targets[known_targets]

            Lx = F.cross_entropy(known_pred,
                                known_targets, reduction='mean')
            loss = Lx
            losses.update(loss.item())
            if len(known_pred) > 0:
                prec1, prec5 = accuracy(known_pred, known_targets, topk=(1, 2))
                top1.update(prec1.item(), known_pred.shape[0])
                top5.update(prec5.item(), known_pred.shape[0])
            
            ind_unk = unk_score > 0.5
            pred_close[ind_unk] = int(outputs.size(1))
            acc_all, unk_acc, size_unk = accuracy_open(pred_close,
                                                       targets,
                                                       num_classes=int(outputs.size(1)))
            acc.update(acc_all.item(), inputs.shape[0])
            unk.update(unk_acc, size_unk)

            batch_time.update(time.time() - end)
            end = time.time()
            if batch_idx == 0:
                unk_all = unk_score
                known_all = known_score
                label_all = targets
                all_out = outputs
            else:
                unk_all = torch.cat([unk_all, unk_score], 0)
                known_all = torch.cat([known_all, known_score], 0)
                label_all = torch.cat([label_all, targets], 0)
                all_out = torch.cat([all_out, outputs], 0)
            
            # Lx = F.cross_entropy(outputs,
                                # targets, reduction='mean')
            # loss = Lx
            # losses.update(loss.item())
            if not args.no_progress:
                test_loader.set_description("Test Iter: {batch:4}/{iter:4}. "
                                            "Data: {data:.3f}s."
                                            "Batch: {bt:.3f}s. "
                                            "Loss: {loss:.4f}. "
                                            "Closed t1: {top1:.3f} "
                                            "t5: {top5:.3f} "
                                            "acc: {acc:.3f}. "
                                            "unk: {unk:.3f}. ".format(
                    batch=batch_idx + 1,
                    iter=len(test_loader),
                    data=data_time.avg,
                    bt=batch_time.avg,
                    loss=losses.avg,
                    top1=top1.avg,
                    top5=top5.avg,
                    acc=acc.avg,
                    unk=unk.avg,
                ))
        if not args.no_progress:
            test_loader.close() 
    # ROC calculation
    unk_all = unk_all.data.cpu().numpy()
    known_all = known_all.data.cpu().numpy()
    label_all = label_all.data.cpu().numpy()
    all_out = all_out.data.cpu().numpy()
    if not val:
        roc, aupr_out, aupr_in= compute_roc_aupr(unk_all, label_all,
                        num_known=int(outputs.size(1)))
        roc_soft = compute_roc(-known_all, label_all,
                            num_known=int(args.num_classes))
        return losses.avg, top1.avg, acc.avg, unk.avg, roc, roc_soft 

    else:
        logger.info("Closed acc: {:.3f}".format(top1.avg))
        return top1.avg


def test_ood(args, test_id, test_loader, model):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    end = time.time()

    if not args.no_progress:
        test_loader = tqdm(test_loader,
                           disable=args.local_rank not in [-1, 0])
    with torch.no_grad():
        for batch_idx, data in enumerate(test_loader):
            if len(data) == 2:
                inputs, targets = data 
            else:
                index, inputs, targets = data
            data_time.update(time.time() - end)
            model.eval()
            inputs = inputs.to(args.device)
            outputs, outputs_open = model(inputs)
            out_open = F.softmax(outputs_open.view(outputs_open.size(0), 2, -1), 1)
            tmp_range = torch.arange(0, out_open.size(0)).long().cuda(args.device)
            pred_close = outputs.data.max(1)[1]
            unk_score = out_open[tmp_range, 0, pred_close]
            batch_time.update(time.time() - end)
            end = time.time()
            if batch_idx == 0:
                unk_all = unk_score
            else:
                unk_all = torch.cat([unk_all, unk_score], 0)
        if not args.no_progress:
            test_loader.close()
    ## ROC calculation
    unk_all = unk_all.data.cpu().numpy()
    roc, aupr_in, aupr_out = misc_id_ood(test_id, unk_all) #aupr_in, aupr_out反了

    return roc, aupr_in, aupr_out
