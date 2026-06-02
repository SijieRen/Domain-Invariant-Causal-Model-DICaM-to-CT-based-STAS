# import torch
from sklearn.metrics import confusion_matrix

def confusion_metrics(predicted_labels, true_labels):
    # print(predicted_labels)
    # _, predicted_labels = torch.max(probs, dim=1)

    # 确保预测的标签是CPU张量并且是整数类型
    predicted_labels = predicted_labels.detach().cpu().numpy()

    # 确保真实标签是CPU张量并且是整数类型
    true_labels = true_labels.detach().cpu().numpy()

    # 使用sklearn的confusion_matrix函数计算混淆矩阵
    conf_matrix = confusion_matrix(true_labels, predicted_labels)

    tn, fp, fn, tp = conf_matrix.ravel()
    sensitivity_test = tp / (tp + fn)
    # print("tn, fp, fn, tp: **********", tn, fp, fn, tp)
    if (tn+fp)!=0:
        specificity_test = tn / (tn + fp)
    else:
        specificity_test = 0

    return sensitivity_test, specificity_test