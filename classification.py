import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold

rfc = RandomForestClassifier(n_estimators=1000)
nb = BernoulliNB()
log_reg = LogisticRegression(solver='liblinear', multi_class='auto')
skf = StratifiedKFold(n_splits=15, shuffle=True, random_state=42)


def plot_conf_matrix(conf_matrix):
    plt.figure(figsize=(5, 5))
    sns.heatmap(conf_matrix, annot=True, cmap="YlGnBu", fmt='g')
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')
    plt.show()


def cross_validation_split(X, y):
    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
    return X_train, X_test, y_train, y_test


def cross_validation_acc(X, y, est):
    X_train, X_test, y_train, y_test = cross_validation_split(X, y)
    scores = cross_val_score(est, X_train, y_train, cv=skf)
    mean_score = sum(scores) / float(len(scores))
    print("Cross Validation Score:", mean_score, "\n")


def check_acc(y_test, pred):
    print("The Accuracy using accuracy_score is", accuracy_score(y_test, pred))
    conf_matrix = confusion_matrix(y_test, pred)
    plot_conf_matrix(conf_matrix)
    print('Classification Report: \n', classification_report(y_test, pred))  # , target_names = seed_names))


def random_forest_classification(X_train, X_test, y_train, y_test):
    rfc.fit(X_train, y_train)
    rfc_pred = rfc.predict(X_test)
    print("Accuracy for Random Forest Classification:\n")
    check_acc(y_test, rfc_pred)


def logistic_regression(X_train, X_test, y_train, y_test):
    StandardScaler().fit_transform(X_train)
    log_reg.fit(X_train, y_train)
    log_pred = log_reg.predict(X_test)
    print("Accuracy for Logistic Regression:\n")
    check_acc(y_test, log_pred)


def naive_bayes(X_train, X_test, y_train, y_test):
    nb.fit(X_train, y_train)
    nbpred = nb.predict(X_test)
    print("Accuracy for Naive Bayes:\n")
    check_acc(y_test, nbpred)


if __name__ == "__main__":
    df = pd.read_csv('C:/Users/cavan/OneDrive/Documents/PL_ML_Predictions/wpl.csv')  # change path if necessary
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    logistic_regression(X_train, X_test, y_train, y_test)
    cross_validation_acc(X, y, log_reg)
    random_forest_classification(X_train, X_test, y_train, y_test)
    cross_validation_acc(X, y, rfc)
    naive_bayes(X_train, X_test, y_train, y_test)
    cross_validation_acc(X, y, nb)





