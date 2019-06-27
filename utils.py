import re
import os
import json
import datetime
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score

def precision_k(actual, predicted, k):
    sort_index = np.argsort(predicted)
    return np.sum(actual[sort_index[-k:]]) / float(k)

def recall_k(actual, predicted, k):
    sort_index = np.argsort(predicted)
    return np.sum(actual[sort_index[-k:]]) / np.sum(actual)

def compute_metrics(ano_scores, label, prefix):

    MAP = average_precision_score(label, ano_scores)
    print(prefix + "-MAP: ", MAP)
    prec_ks = []
    recall_ks = []
    k = 2000
    for i in range(k):
        prec = precision_k(label, ano_scores, i+1)
        prec_ks.append(prec)
        recall = recall_k(label, ano_scores, i+1)
        recall_ks.append(recall)

    metrics = {"MAP": MAP,
               "prec@k": prec_ks,
               "recall@k": recall_ks}

    # save metrics
    ind = 0
    existings = re.findall(prefix+"-\d+", "\t".join(os.listdir("metrics")))
    if existings:
        existings = sorted(existings, key=lambda x: int(x.split("-")[-1]))
        ind = int(existings[-1].split("-")[-1]) + 1

    w = 2
    if "weight" in os.environ:
        w = os.environ["weight"]

    with open("metrics/{}-{}-w{}.json".format(prefix, ind, w), "w") as fw:
        fw.write(json.dumps(metrics, indent=4))