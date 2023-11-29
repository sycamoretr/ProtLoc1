#多标签评估指标
from sklearn.metrics import f1_score
from sklearn.metrics import hamming_loss,jaccard_score
from sklearn.metrics import accuracy_score
import numpy as np



def Average_Precision(label, logit):
    N = len(label)
    for i in range(N):
        if max(label[i]) == 0 or min(label[i]) == 1:
            print("该条数据哪一类都不是或者全都是")
    precision = 0
    for i in range(N):
        index = np.where(label[i] == 1)[0]
        score = logit[i][index]
        score = sorted(score)
        score_all = sorted(logit[i])
        precision_tmp = 0
        for item in score:
            tmp1 = score.index(item)
            tmp1 = len(score) - tmp1
            tmp2 = score_all.index(item)
            tmp2 = len(score_all) - tmp2
            precision_tmp += tmp1 / tmp2
        precision += precision_tmp / len(score)
    Average_Precision = precision / N
    return Average_Precision

def Metrics(y_score,y_true):
    y_true = np.array(y_true)
    y_score = np.array(y_score)
    y_pred = np.around(y_score)
    ACC = accuracy_score(y_true, y_pred)
    AP = Average_Precision(y_true,y_score)
    HL = hamming_loss(y_true, y_pred)
    F1 = f1_score(y_true,y_pred,average='samples')
    jaccard = jaccard_score(y_true, y_pred, average='samples')
    return ACC,AP, HL,F1,jaccard











