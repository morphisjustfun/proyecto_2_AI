import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.svm import SVC
from sklearn.metrics import precision_score, recall_score, f1_score, roc_curve, auc
from sklearn.preprocessing import label_binarize
import subprocess
import matplotlib.pyplot as plt


def roc_curve_no_prob(y_true, y_pred, k):
    fpr = []
    tpr = []

    for threshold in range(k):
        y_pred_k = y_pred[threshold]

        fp = np.sum((y_pred_k == 1) & (y_true == 0))
        tp = np.sum((y_pred_k == 1) & (y_true == 1))

        fn = np.sum((y_pred_k == 0) & (y_true == 1))
        tn = np.sum((y_pred_k == 0) & (y_true == 0))

        fpr.append(fp / (fp + tn))
        tpr.append(tp / (tp + fn))

    return [fpr, tpr]


def get_scores(method, filename='data/data.csv', label='label', k=4, split='mean'):
    # method can be kfold or boostraping
    if method == 'kfold':
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

            precision, recall, f1, auc = test_dstree(
                fold, df_test, label, split, False)
            df_results = df_results.append(
                {'model': 'svm', 'fold': fold, 'precision': precision,
                 'recall': recall, 'f1': f1, 'auc': auc},
                ignore_index=True)

            precision, recall, f1, auc = test_kdtree(
                fold, df_test, label, k, False)
            df_results = df_results.append(
                {'model': 'kdtree', 'fold': fold, 'precision': precision,
                    'recall': recall, 'f1': f1, 'auc': auc},
                ignore_index=True)

            precision, recall, f1, auc = test_svm(
                fold, df_train, df_test, label, False)
            df_results = df_results.append(
                {'model': 'dstree', 'fold': fold, 'precision': precision,
                    'recall': recall, 'f1': f1, 'auc': auc},
                ignore_index=True)
            fold += 1

        df_results.to_csv('data/results/kfold_results_folds.csv', index=False)

    if method == 'bootstrapping':
        df = pd.read_csv(filename)
        df_results = pd.DataFrame(
            columns=['model', 'fold', 'precision', 'recall', 'f1', 'auc'])
        for fold in range(3):
            df_train = df.sample(frac=0.8, replace=True)
            df_test = df.drop(df_train.index)
            df_train.to_csv('data/train.csv', index=False)
            df_test.to_csv('data/test.csv', index=False)

            precision, recall, f1, auc = test_dstree(
                fold, df_test, label, split, True)
            df_results = df_results.append(
                {'model': 'svm', 'fold': fold, 'precision': precision,
                 'recall': recall, 'f1': f1, 'auc': auc},
                ignore_index=True)

            precision, recall, f1, auc = test_kdtree(
                fold, df_test, label, k, True)
            df_results = df_results.append(
                {'model': 'kdtree', 'fold': fold, 'precision': precision,
                    'recall': recall, 'f1': f1, 'auc': auc},
                ignore_index=True)

            precision, recall, f1, auc = test_svm(
                fold, df_train, df_test, label, True)
            df_results = df_results.append(
                {'model': 'dstree', 'fold': fold, 'precision': precision,
                    'recall': recall, 'f1': f1, 'auc': auc},
                ignore_index=True)

        df_results.to_csv('data/results/bs_results_folds.csv', index=False)


def test_kdtree(fold, df_test, label, k, bs):
    subprocess.run(
        ['../C++/cmake-build-debug/proyecto_2', 'kdtree', str(k)])
    y_test = df_test[label]
    y_pred_k = pd.read_csv(f'data/kdtree_results_k{k}.csv')['label'].to_numpy()
    precision_k = precision_score(y_test, y_pred_k, average='weighted')
    recall_k = recall_score(y_test, y_pred_k, average='weighted')
    f1_k = f1_score(y_test, y_pred_k, average='weighted')
    n_classes = df_test[label].nunique()

    y_pred_pre_k = []
    for value_k in range(1, k):
        y_pred = pd.read_csv(
            f'data/kdtree_results_k{value_k}.csv')['label'].to_numpy()
        y_pred_pre_k.append(y_pred)
    y_pred_all = np.array(y_pred_pre_k + [y_pred_k])
    y_test_bin = label_binarize(y_test, classes=range(n_classes))
    y_pred_all_bin = []
    for y_pred in y_pred_all:
        y_pred_bin = label_binarize(y_pred, classes=range(n_classes))
        y_pred_all_bin.append(y_pred_bin)

    fpr = dict()
    tpr = dict()
    for n_class in range(n_classes):
        y_pred_all_bin_class = [y_pred_all_bin[i][:, n_class]
                                for i in range(len(y_pred_all_bin))]
        fpr[n_class], tpr[n_class] = roc_curve_no_prob(y_test_bin[:, n_class],
                                                       y_pred_all_bin_class,
                                                       k)

    fpr["micro"] = np.mean([fpr[i] for i in range(n_classes)], axis=0)
    tpr["micro"] = np.mean([tpr[i] for i in range(n_classes)], axis=0)

    fpr_tpr = dict(zip(fpr["micro"], tpr["micro"]))
    fpr_tpr = {k: v for k, v in sorted(
        fpr_tpr.items(), key=lambda item: item[0])}
    fpr["micro"] = list(fpr_tpr.keys())
    tpr["micro"] = list(fpr_tpr.values())

    fpr["micro"] = np.insert(fpr["micro"], 0, 0)
    fpr["micro"] = np.append(fpr["micro"], 1)
    tpr["micro"] = np.insert(tpr["micro"], 0, 0)
    tpr["micro"] = np.append(tpr["micro"], 1)

    auc_score = auc(fpr["micro"], tpr["micro"])

    plt.clf()
    plt.figure()
    plt.plot(fpr["micro"], tpr["micro"])
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Tasa de Falsos Positivos')
    plt.ylabel('Tasa de Verdaderos Positivos')
    plt.title(f'Curva ROC - KNN - Fold {fold}')
    plt.savefig(
        f'data/plots/{"bs" if bs else "kfold"}_roc_kdtree_fold{fold}.png')

    return precision_k, recall_k, f1_k, auc_score


def test_dstree(fold, df_test, label, split, bs):
    splits = ['mean', 'median']
    splits.remove(split)
    y_test = df_test[label]

    subprocess.run(
        ['../C++/cmake-build-debug/proyecto_2', 'ds_tree', str(split)])
    y_pred_s_split = pd.read_csv(
        f'data/ds_tree_results_{split}.csv')['label'].to_numpy()

    precision_s_split = precision_score(
        y_test, y_pred_s_split, average='weighted')
    recall_s_split = recall_score(y_test, y_pred_s_split, average='weighted')
    f1_s_split = f1_score(y_test, y_pred_s_split, average='weighted')

    y_pred_pre_s_split = []
    for remaining_split in splits:
        subprocess.run(
            ['../C++/cmake-build-debug/proyecto_2', 'ds_tree', str(remaining_split)])
        y_pred = pd.read_csv(
            f'data/ds_tree_results_{remaining_split}.csv')['label'].to_numpy()
        y_pred_pre_s_split.append(y_pred)

    y_pred_all = np.array(y_pred_pre_s_split + [y_pred_s_split])
    n_classes = df_test[label].nunique()
    y_test_bin = label_binarize(y_test, classes=range(n_classes))
    y_pred_all_bin = []
    for y_pred in y_pred_all:
        y_pred_bin = label_binarize(y_pred, classes=range(n_classes))
        y_pred_all_bin.append(y_pred_bin)

    fpr = dict()
    tpr = dict()
    for n_class in range(n_classes):
        y_pred_all_bin_class = [y_pred_all_bin[i][:, n_class]
                                for i in range(len(y_pred_all_bin))]
        fpr[n_class], tpr[n_class] = roc_curve_no_prob(y_test_bin[:, n_class],
                                                       y_pred_all_bin_class,
                                                       len(splits) + 1)

    fpr["micro"] = np.mean([fpr[i] for i in range(n_classes)], axis=0)
    tpr["micro"] = np.mean([tpr[i] for i in range(n_classes)], axis=0)

    fpr_tpr = dict(zip(fpr["micro"], tpr["micro"]))
    fpr_tpr = {k: v for k, v in sorted(
        fpr_tpr.items(), key=lambda item: item[0])}
    fpr["micro"] = list(fpr_tpr.keys())
    tpr["micro"] = list(fpr_tpr.values())

    fpr["micro"] = np.insert(fpr["micro"], 0, 0)
    fpr["micro"] = np.append(fpr["micro"], 1)
    tpr["micro"] = np.insert(tpr["micro"], 0, 0)
    tpr["micro"] = np.append(tpr["micro"], 1)

    # sort tpr["micro"] and preserve that order in fpr["micro"]

    auc_score = auc(fpr["micro"], tpr["micro"])
    plt.clf()
    plt.figure()
    plt.plot(fpr["micro"], tpr["micro"])
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Tasa de Falsos Positivos')
    plt.ylabel('Tasa de Verdaderos Positivos')
    plt.title(f'Curva ROC - DSTree - Fold {fold}')
    plt.savefig(
        f'data/plots/{"bs" if bs else "kfold"}_roc_dstree_fold{fold}.png')
    return precision_s_split, recall_s_split, f1_s_split, auc_score


def test_svm(fold, df_train, df_test, label, bs):
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
        fpr[n_class], tpr[n_class], _ = roc_curve(
            y_test_bin[:, n_class], y_pred_prob[:, n_class])
        roc_auc[n_class] = auc(fpr[n_class], tpr[n_class])

    fpr["micro"], tpr["micro"], _ = roc_curve(
        y_test_bin.ravel(), y_pred_prob.ravel())
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
    plt.title(f'Curva ROC - SVM - Fold {fold}')
    plt.savefig(
        f'data/plots/{"bs" if bs else "kfold"}_roc_svm_fold_{fold}.png')
    return precision, recall, f1, roc_auc["micro"]


if __name__ == '__main__':
    get_scores('kfold')
    get_scores('bootstrapping')
