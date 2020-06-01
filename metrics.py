#!/usr/bin/env python
# coding=utf-8
"""
Calculate the AUC, sensitivity, specificity
and F1-score
"""
from sklearn import metrics
from sklearn.metrics import roc_curve, auc, average_precision_score
from sklearn.metrics import confusion_matrix, precision_recall_curve
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


GT_CSV_DIR = "./statistic_description/tmp/test.csv"
PRED_CSV_DIR = "./tmp/detection_results/cls.csv"
GT_DF = pd.read_csv(GT_CSV_DIR, na_filter=False)
PRED_DF = pd.read_csv(PRED_CSV_DIR, na_filter=False)
GT_DF["label"] = GT_DF["annotation"].map(lambda x: (x=="" and "0")
                                         or (x!="" and "1"))
Y_TRUE = list(GT_DF["label"])
Y_TRUE = [int(i) for i in Y_TRUE]
Y_PRED = list(PRED_DF["prediction"])


def find_optimal_cutoff(target, predicted):
    fpr, tpr, threshold = roc_curve(target, predicted)
    i = np.arange(len(tpr))
    roc = pd.DataFrame({"tf": pd.Series(tpr-(1-fpr), index=i),
                        "threshold": pd.Series(threshold, index=i)})
    roc_t = roc.iloc[(roc.tf - 0).abs().argsort()[:1]]
    return list(roc_t["threshold"])


if __name__ == "__main__":
    fpr, tpr, threshold = roc_curve(Y_TRUE, Y_PRED)
    roc_auc = auc(fpr, tpr)
    average_precision = average_precision_score(Y_TRUE, Y_PRED)
    precision, recall, _ = precision_recall_curve(Y_TRUE, Y_PRED)
    #===========plot ROC Curve===================================
    # fig, ax = plt.subplots(
    #     subplot_kw=dict(xlim=[0, 1], ylim=[0, 1.01], aspect="equal"),
    #     figsize=(6, 6)
    # )
    # ax.plot(fpr, tpr, label=f'AUC: {roc_auc:.03}')
    # # ax.scatter(y=566/(566+92), x=(1-513/(85+513)), s=10,
    # #            color="#dd2c5f")
    # # ax.plot([(1-513/(85+513)), 1-513/(513+85)], [0, 566/(566+92)])
    # # ax.plot([0, 1-513/(513+85)], [566/(566+92), 566/(566+92)])
    
    # _ = ax.legend(loc="lower right")
    # ax.set_xlabel("False Positive Rate")
    # ax.set_ylabel("True Positive Rate")
    # ax.grid(linestyle="dashed")
    # fig.savefig("roc.svg", dpi=300, format="svg")
    
    # thresh= find_optimal_cutoff(Y_TRUE, Y_PRED)
    # PRED_DF["pred_label"] = PRED_DF["prediction"].map(
    #     lambda x: 1 if x>=thresh[0] else 0
    # )
    # print(confusion_matrix(Y_TRUE, PRED_DF["pred_label"]))
    #===============================================================
    #========plot PR Curve=========================================
    # fig, ax = plt.subplots(
    #     subplot_kw=dict(xlim=[0, 1], ylim=[0, 1.01], aspect="equal"),
    #     figsize=(6,6)
    # )
    # ax.plot(recall, precision, label=f'AP: {average_precision:.03}')
    # ax.set_xlabel("Recall")
    # ax.set_ylabel("Precision")
    # ax.grid(linestyle="dashed")
    # f_scores = np.linspace(0.2, 0.8, num=4)
    # for i, f_score in enumerate(f_scores):
    #     x = np.linspace(0.01, 1)
    #     y = f_score * x / (2*x-f_score)
    #     l, = ax.plot(x[y>=0], y[y>=0], color="gray", alpha=0.2,
    #                  linewidth=1, label="iso-f1 curve" if i==3 else None)
    #     ax.annotate("f1={0:0.1f}".format(f_score),
    #                 xy=(0.8, y[45]+0.02))
    # _ = ax.legend(loc="best")
    # fig.savefig("pr.svg", dpi=300, format="svg")
    #==============================================================
    #========plot detection PR curve===============================
    from tools.voc_eval_new import custom_voc_eval

    gt_csv = "./statistic_description/tmp/test.csv"
    pred_csv = "./tmp/detection_results/loc.csv"

    rec, prec, ap = custom_voc_eval(gt_csv, pred_csv)
    fig, ax = plt.subplots(
        subplot_kw=dict(xlim=[0, 1], ylim=[0, 1.01], aspect="equal"),
        figsize=(6,6)
    )
    ax.plot(rec, prec, label=f'AP: {ap:.03}')
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.grid(linestyle="dashed")
    _ = ax.legend(loc="lower right")
    fig.savefig("det_pr.svg", dpi=300, format="svg")
    



