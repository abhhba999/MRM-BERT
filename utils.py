import pandas as pd
import os
from sklearn.metrics import roc_auc_score
import numpy as np
def generate_list(test_pred_list, test_prob_list):
    n = len(test_pred_list[0])  # 获取列表长度
    result_list = []
    for i in range(n):
        count = 0
        for j in range(5):
            if test_pred_list[j][i] == 1:
                count += 1
        if count >= 3:
            result_list.append(1)
        else:
            result_list.append(0)

    list2 = []
    for i in range(n):
        count = 0
        for j in range(5):
            if test_pred_list[j][i] == result_list[i]:
                count += 1
        avg_prob = sum(test_prob_list[j][i] for j in range(5) if test_pred_list[j][i] == result_list[i]) / count
        list2.append(avg_prob)

    return result_list, list2
 
#---自己按照公式实现
def auc_calculate(labels,preds,n_bins=100):
    postive_len = sum(labels)
    negative_len = len(labels) - postive_len
    total_case = postive_len * negative_len
    pos_histogram = [0 for _ in range(n_bins)]
    neg_histogram = [0 for _ in range(n_bins)]
    bin_width = 1.0 / n_bins
    for i in range(len(labels)):
        nth_bin = int(preds[i]/bin_width)
        if labels[i]==1:
            pos_histogram[nth_bin] += 1
        else:
            neg_histogram[nth_bin] += 1
    accumulated_neg = 0
    satisfied_pair = 0
    for i in range(n_bins):
        satisfied_pair += (pos_histogram[i]*accumulated_neg + pos_histogram[i]*neg_histogram[i]*0.5)
        accumulated_neg += neg_histogram[i]
 
    return satisfied_pair / float(total_case)
def read_fasta(file_path):
    with open(file_path, 'r') as file:
        labels = []
        sequences = []
        sequence = ''
        for line in file:
            line = line.strip()
            if line.startswith('>'):
                labels.append(int(line[3:]))
                if sequence != '':
                    sequences.append(sequence)
                    sequence = ''
            else:
                sequence += line
        sequences.append(sequence)  # add the last sequence

    data = { 'sequence': sequences,'label': labels}
    df = pd.DataFrame(data)

    return df
def read_fasta_all(file_path):
    with open(file_path, 'r') as file:
        labels = []
        sequences = []
        sequence = ''
        for line in file:
            line = line.strip()
            if line.startswith('>'):
                labels.append(int(line.split('|')[1]))
                if sequence != '':
                    sequences.append(sequence)
                    sequence = ''
            else:
                sequence += line
        sequences.append(sequence)  # add the last sequence

    data = { 'sequence': sequences,'label': labels}
    df = pd.DataFrame(data)

    return df
def read_data(a, b):
    # 拼接文件路径
    folder_path = os.path.join(a, b)
    pos_file_path = os.path.join(folder_path, 'positive.txt')
    neg_file_path = os.path.join(folder_path, 'negative.txt')

    # 读取正样本和负样本文件，生成DataFrame
    pos_df = pd.read_csv(pos_file_path, header=None, names=['sequence'])
    pos_df['label'] = 1
    neg_df = pd.read_csv(neg_file_path, header=None, names=['sequence'])
    neg_df['label'] = 0
    df = pd.concat([pos_df, neg_df], ignore_index=True)
    return df
def matrix(predictions,true_labels):
    # 计算真正数（Truefor Positive，TP）、假真数（False Positive，FP）、假负数（False Negative，FN）和真负数（True Negative，TN）的数量
    TP = sum([1 for i in range(len(predictions)) if predictions[i] == 1 and true_labels[i] == 1])
    FP = sum([1 for i in range(len(predictions)) if predictions[i] == 1 and true_labels[i] == 0])
    FN = sum([1 for i in range(len(predictions)) if predictions[i] == 0 and true_labels[i] == 1])
    TN = sum([1 for i in range(len(predictions)) if predictions[i] == 0 and true_labels[i] == 0])


    # 计算各个指标
    ACC = (TP + TN) / len(predictions)
    Sn = TP / (TP + FN)
    Sp = TN / (TN + FP)
    if(((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))**0.5==0):
        MCC=0
    else:
        MCC=(TP*TN-FP*FN)/((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))**0.5

    # 打印结果
    print("ACC: {:.2f}%".format(ACC * 100))
    print("Sn: {:.2f}%".format(Sn * 100))
    print("Sp: {:.2f}%".format(Sp * 100))
    print("MCC: {:.2f}%".format(MCC * 100))
    return [ACC,Sn,Sp,MCC]
from sklearn.metrics import matthews_corrcoef, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score
def simple_accuracy(preds, labels):
    return (preds == labels).mean()
def matrix2(preds, labels, probs):
    acc=simple_accuracy(preds, labels)
    precision = precision_score(y_true=labels, y_pred=preds)
    recall = recall_score(y_true=labels, y_pred=preds)
    f1 = f1_score(y_true=labels, y_pred=preds)
    mcc = matthews_corrcoef(labels, preds)
    auc = roc_auc_score(labels, probs)
    aupr = average_precision_score(labels, probs)
    TP = sum([1 for i in range(len(preds)) if preds[i] == 1 and labels[i] == 1])
    FP = sum([1 for i in range(len(preds)) if preds[i] == 1 and labels[i] == 0])
    FN = sum([1 for i in range(len(preds)) if preds[i] == 0 and labels[i] == 1])
    TN = sum([1 for i in range(len(preds)) if preds[i] == 0 and labels[i] == 0])


    # 计算各个指标
    ACC = (TP + TN) / len(preds)
    Sn = TP / (TP + FN)
    Sp = TN / (TN + FP)
    if(((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))**0.5==0):
        MCC=0
    else:
        MCC=(TP*TN-FP*FN)/((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))**0.5
    

    # 打印结果
    print("ACC: {:.4f}%".format(ACC * 100))
    print("Sn : {:.4f}%".format(Sn * 100))
    print("Sp : {:.4f}%".format(Sp * 100))
    print("MCC: {:.4f}%".format(MCC * 100))
    print(
        "F1 :",f1,
        "\nAUC:", auc,
        "\nprecision:",precision,
        "\nrecall:",recall
    )
    return [ACC,Sn,Sp,MCC,auc,f1]
           #str({"Acc":ACC,"Sn":Sn,"Sp":Sp,"MCC":MCC,"AUC":auc,"F1":f1})
def seq2kmer(seq):
    k=3
    kmer = [seq[x:x+k] for x in range(len(seq)+1-k)]
    kmers = " ".join(kmer)
    return kmers