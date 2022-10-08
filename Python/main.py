import warnings

warnings.filterwarnings("ignore")

import pandas as pd
from sklearn.model_selection import KFold
from sklearn.svm import SVC
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, plot_roc_curve, roc_curve, auc
from sklearn.preprocessing import label_binarize
import subprocess
import matplotlib.pyplot as plt
import numpy as np


def get_scores(filename='data/data.csv', label='label', k=3, split='mean'):
    df = pd.read_csv(filename)
    kfold = KFold(n_splits=3)
    df_results = pd.DataFrame(
        columns=['model', 'fold', 'precision', 'recall', 'f1', 'auc'])
    fold = 0
    for train_index, test_index in kfold.split(df):
        df_train = df.iloc[train_index]
        df_test = df.iloc[test_index]
        df_train.to_csv('data/train.csv', index=False)
        df_test.to_csv('data/test.csv', index=False)
        precision, recall, f1 = test_kdtree(fold, df_test, label, k)
        df_results = df_results.append(
            {'model': 'kdtree', 'fold': fold, 'precision': precision, 'recall': recall, 'f1': f1}, ignore_index=True)
        precision, recall, f1 = test_dstree(fold, df_test, label, split)
        df_results = df_results.append(
            {'model': 'dstree', 'fold': fold, 'precision': precision, 'recall': recall, 'f1': f1}, ignore_index=True)
        precision, recall, f1, auc = test_svm(fold, df_train, df_test, label)
        df_results = df_results.append(
            {'model': 'svm', 'fold': fold, 'precision': precision,
             'recall': recall, 'f1': f1, 'auc': auc},
            ignore_index=True)
        fold += 1
    df_results.to_csv('data/results_folds.csv', index=False)

    df_results_summarized = df_results.groupby('model').mean()
    df_results_summarized.drop(['fold'], axis=1, inplace=True)
    df_results_summarized.reset_index(inplace=True)
    df_results_summarized.to_csv('data/results_summarized.csv', index=False)


def test_kdtree(fold, df_test, label, k):
    subprocess.run(
        ['../C++/cmake-build-debug/proyecto_2', 'kdtree', str(k)])
    y_test = df_test[label]
    y_pred = pd.read_csv('data/kdtree_results.csv')['label'].to_numpy()
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    return precision, recall, f1


def test_dstree(fold, df_test, label, split):
    subprocess.run(
        ['../C++/cmake-build-debug/proyecto_2', 'ds_tree', str(split)])
    y_test = df_test[label]
    y_pred = pd.read_csv('data/ds_tree_results.csv')['label'].to_numpy()
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    return precision, recall, f1


def test_svm(fold, df_train, df_test, label):
    # test support vector machine
    x_train = df_train.drop(columns=[label])
    y_train = df_train[label]
    x_test = df_test.drop(columns=[label])
    y_test = df_test[label]
    clf = SVC(probability=True)
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    y_pred_prob = clf.predict_proba(x_test)
    # multiclass
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

    n_classes = y_train.nunique()
    y_test_bin = label_binarize(y_test, classes=range(n_classes))

    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for n_class in range(n_classes):
        fpr[n_class], tpr[n_class], _ = roc_curve(y_test_bin[:, n_class], y_pred_prob[:, n_class])
        roc_auc[n_class] = auc(fpr[n_class], tpr[n_class])

    fpr["micro"], tpr["micro"], _ = roc_curve(y_test_bin.ravel(), y_pred_prob.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    plt.clf()
    plt.figure()
    plt.plot(fpr["micro"], tpr["micro"],
             label='Curva ROC promedio (área = {0:0.2f})'
                   ''.format(roc_auc["micro"]))

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Tasa de Falsos Positivos')
    plt.ylabel('Tasa de Verdaderos Positivos')
    plt.title('Curva ROC')
    plt.savefig(f'data/plots/roc_svm_fold_{fold}.png')
    return precision, recall, f1, roc_auc["micro"]


if __name__ == '__main__':
    get_scores()
