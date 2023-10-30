import numpy as np
from sklearn.mixture import GaussianMixture
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from utils import ova_ent, compute_roc,compute_S_ID
from sklearn.metrics import roc_auc_score, average_precision_score, recall_score


def get_sequence(args, tmp_data, query_num):
    ood_list = []
    tmp_seq = np.argsort(tmp_data)[0][:query_num]
    Un_Value = tmp_data[0]
    query_Index = tmp_data[1]
    labelArr = tmp_data[2]
    
    Un_Value = [Un_Value[i] for i in tmp_seq]
    query_Index = [query_Index[i] for i in tmp_seq]
    query_Index = list(map(int, query_Index))
    labelArr = [labelArr[i] for i in tmp_seq]

    for i in range(len(labelArr)):
        if labelArr[i] >= args.num_classes:
            ood_list.append(int(query_Index[i]))
    
    return ood_list, query_Index


def get_sequence_back(args, tmp_data, query_num):
    ood_list = []
    tmp_seq = np.argsort(tmp_data)[0][-query_num:]
    Un_Value = tmp_data[0]
    query_Index = tmp_data[1]
    labelArr = tmp_data[2]
    
    Un_Value = [Un_Value[i] for i in tmp_seq]
    query_Index = [query_Index[i] for i in tmp_seq]
    query_Index = list(map(int, query_Index))
    labelArr = [labelArr[i] for i in tmp_seq]

    for i in range(len(labelArr)):
        if labelArr[i] >= args.num_classes:
            ood_list.append(int(query_Index[i]))
    
    return ood_list, query_Index



def Compute_un(args, unlabeledloader, Len_labeled_ind_train, model, wnet, query, ID_need, miu):
    print('-'*40 + ' Start Sampling ' + '-'*40)
    OOD_need = args.query_batch - ID_need
    wnet.eval()
    model.eval()

    temp_label = []
    Un_Value = []
    queryIndex = []
    labelArr = []
    ood_list = []
    for batch_idx, data in enumerate(unlabeledloader):
        fea, labels, index = data
        labels = labels.tolist()
        labels = torch.Tensor(labels)
        inputs = fea
        index = index.cpu().tolist()
        if args.device:
            inputs, labels = inputs.to(args.device), labels.to(args.device)
        outputs, outputs_open = model(inputs)

        # get the predicted label of label
        batch_size = inputs.shape[0]
        v_ij, predicted = outputs.max(1)
        # 使用ova的预测作为
        predicted = predicted
        weight = wnet(outputs_open).cpu()
        s_ID = compute_S_ID(outputs_open)
        s_ID = torch.reshape(s_ID, (len(s_ID),1)).cpu()
        # else:
            # s_ID = torch.reshape(1-s_ID, (len(s_ID),1))
        Un = weight + miu*s_ID
        Un = Un.squeeze(1)
        Un = Un.cpu().tolist()

        if query > 0:
            tmp_lab = predicted.cpu().tolist()
            weight_list = weight.detach().numpy().tolist()
            tmp_sd = s_ID.tolist()
            # print(tmp_lab)
            # print()
            for ind, lab in enumerate(tmp_lab):
                if lab == args.num_classes:
                    un_val = weight_list[ind][0] + miu*(1.0-tmp_sd[ind][0])
                    Un[ind] = un_val
        temp_label += predicted.cpu()
        Un_Value += Un
        queryIndex += index
        labelArr += list(np.array(labels.cpu()))
    
    tmp_data = np.vstack((Un_Value, queryIndex, labelArr, temp_label))

    Unlabel_ID = len(np.where(np.array(labelArr) < args.num_classes)[0])

    if query == 0:
        tmp_seq = np.argsort(tmp_data)[0][:args.query_batch]
        Un_Value = tmp_data[0]
        query_Index = tmp_data[1]
        labelArr = tmp_data[2]
        temp_Label = tmp_data[3]
        Un_Value = [Un_Value[i] for i in tmp_seq]
        query_Index = [query_Index[i] for i in tmp_seq]
        query_Index = list(map(int, query_Index))
        labelArr = [labelArr[i] for i in tmp_seq]
        for i in range(len(labelArr)):
            if labelArr[i] >= args.num_classes:
                ood_list.append(int(query_Index[i]))
        
        ood_beg, q_beg = get_sequence(args, tmp_data, args.query_batch-300)
        ood_back, q_back = get_sequence_back(args, tmp_data, 300)


        prec_back = (len(ood_back) + len(ood_beg)) / (len(q_back) + len(q_beg))

        select_ID = list(set(q_back)-set(ood_back)) + list(set(q_beg) - set(ood_beg))

        select_ID_len = len(select_ID)
        ood_back_len = len(ood_back) + len(ood_beg)
        precision = select_ID_len / 1500
        recall = (Len_labeled_ind_train + select_ID_len)/(Len_labeled_ind_train + Unlabel_ID)
        quey_back = q_beg + q_back
        print('1500 ood_numer precision is:' + str(prec_back))
        print('1500 ood_numer is:' + str(int(ood_back_len)))
        print('-'*40 + ' Finished Sampling ' + '-'*40)
        return select_ID, quey_back, precision, recall
    
    else:
        ood_back, q_back = get_sequence_back(args, tmp_data, args.query_batch)
        prec_back = len(ood_back) / len(q_back)
        print('back 1500 ood_numer precision is:' + str(prec_back))
        Un_Value = tmp_data[0]
        query_Index = tmp_data[1]
        labelArr = tmp_data[2]
        temp_Label = tmp_data[3]

        targets_unk = temp_Label >= int(args.num_classes)
        targets_know = temp_Label < int(args.num_classes)
    
        ID_tmps = np.vstack((Un_Value[targets_know],query_Index[targets_know],labelArr[targets_know],temp_Label[targets_know]))
        OOD_tmps = np.vstack((Un_Value[targets_unk],query_Index[targets_unk],labelArr[targets_unk],temp_Label[targets_unk]))
        print('the pseudo label of ID is '+str(len(ID_tmps[0])))
        print('the pseudo label of OOD is '+str(len(OOD_tmps[0])))

        ood_back_ID, ID_back = get_sequence_back(args, ID_tmps, ID_need)
        ID_ID_back = list(set(ID_back)-set(ood_back_ID))
        prec_back_ID = len(ood_back_ID) / len(ID_back)
        print('Last OOD ood_numer precision is:' + str(prec_back_ID) + " and the number of OOD is " + str(len(ood_back_ID))+ " and the number of ID is " + str(len(ID_ID_back)))

        if OOD_need > 0:
            if len(OOD_tmps[0]) >= OOD_need:
                ood_back_OOD, OOD_back = get_sequence(args, OOD_tmps, OOD_need)
                OOD_ID_back = list(set(OOD_back)-set(ood_back_OOD))
                prec_back_OOD = len(ood_back_OOD) / len(OOD_back)
            elif len(OOD_tmps[0]) > 0:
                ood_back_OOD, OOD_back = get_sequence(args, OOD_tmps, len(OOD_tmps[0]))
                OOD_ID_back = list(set(OOD_back)-set(ood_back_OOD))
                prec_back_OOD = len(ood_back_OOD) / len(OOD_back)
            else:
                OOD_ID_back = []
                OOD_back = []
                prec_back_OOD = 0
                ood_back_OOD = []
            print('OOD ood_numer precision is:' + str(prec_back_OOD) + " and the number of OOD is " + str(len(ood_back_OOD)))
        else:
            OOD_back = []
            ood_back_OOD = []
            OOD_ID_back = []
        quey_back = ID_back + OOD_back

        precision = len(ID_ID_back) / int(ID_need)
        recall = (Len_labeled_ind_train + len(ID_ID_back))/(Len_labeled_ind_train + Unlabel_ID + len(OOD_ID_back))
        print('-'*40 + ' Finished Sampling ' + '-'*40)
        # return ID_ID, quey_first, precision, recall
        return ID_ID_back, quey_back, precision, recall



