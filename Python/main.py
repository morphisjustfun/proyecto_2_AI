import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.svm import SVC
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
import subprocess


def get_scores(filename='data/data.csv', label='label', k=3, split='mean'):
    df = pd.read_csv(filename)
    kfold = KFold(n_splits=3)
    df_results = pd.DataFrame(columns=['model', 'fold', 'precision', 'recall', 'f1', 'auc'])
    fold = 0
    for train_index, test_index in kfold.split(df):
        df_train = df.iloc[train_index]
        df_test = df.iloc[test_index]
        df_train.to_csv('data/train.csv', index=False)
        df_test.to_csv('data/test.csv', index=False)
        precision, recall, f1 = test_kdtree(df_test, label, k)
        df_results = df_results.append(
            {'model': 'kdtree', 'fold': fold, 'precision': precision, 'recall': recall, 'f1': f1}, ignore_index=True)
        precision, recall, f1 = test_dstree(df_test, label, split)
        df_results = df_results.append(
            {'model': 'dstree', 'fold': fold, 'precision': precision, 'recall': recall, 'f1': f1}, ignore_index=True)
        precision, recall, f1, auc = test_svm(df_train, df_test, label)
        df_results = df_results.append(
            {'model': 'svm', 'fold': fold, 'precision': precision, 'recall': recall, 'f1': f1, 'auc': auc},
            ignore_index=True)
        fold += 1
    df_results.to_csv('data/results_folds.csv', index=False)
    df_results = df_results.groupby('model').mean()
    df_results.to_csv('data/results_summarized.csv', index=False)


def test_svm(df_train, df_test, label):
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
    auc = roc_auc_score(y_test, y_pred_prob, average='weighted', multi_class='ovr')
    return precision, recall, f1, auc


def test_kdtree(df_test, label, k):
    subprocess.run(['../C++/cmake-build-debug/proyecto_2', 'kdtree', k])
    y_test = df_test[label]
    y_pred = pd.read_csv('data/kdtree_results.csv')['label'].to_numpy()
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    return precision, recall, f1


def test_dstree(df_test, label, split):
    subprocess.run(['../C++/cmake-build-debug/proyecto_2', 'dstree', split])
    y_test = df_test[label]
    y_pred = pd.read_csv('data/dstree_results.csv')['label'].to_numpy()
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    return precision, recall, f1


if __name__ == '__main__':
    get_scores()
