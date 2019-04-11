import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import BernoulliNB, MultinomialNB, GaussianNB, ComplementNB
from sklearn.model_selection import train_test_split, cross_val_score, TimeSeriesSplit
from xgboost import XGBClassifier

rfc = RandomForestClassifier(n_estimators=500)
bnb = BernoulliNB()
mnb = MultinomialNB()
gnb = GaussianNB()
cnb = ComplementNB()
log_reg = LogisticRegression(solver='liblinear', multi_class='auto')
s_log_reg = LogisticRegression(solver='liblinear', multi_class='auto')
tscv = TimeSeriesSplit(n_splits=10)
xgb = XGBClassifier()


def plot_conf_matrix(conf_matrix):
    plt.figure(figsize=(7, 7))
    sns.heatmap(conf_matrix, annot=True, cmap="YlGnBu", fmt='g')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.show()


def plot_bar_chart():
    plt.bar()


def check_acc(y_test, pred):
    print("The Accuracy using accuracy_score is", accuracy_score(y_test, pred))
    conf_matrix = confusion_matrix(y_test, pred)
    plot_conf_matrix(conf_matrix)
    print('Classification Report:\n', classification_report(y_test, pred))  # , target_names = seed_names))


def logistic_regression():
    log_reg.fit(X_train, y_train)
    #s_log_reg.fit(Xs_train, ys_train)
    log_pred = log_reg.predict(X_test)
    #s_log_reg_pred = s_log_reg.predict(Xs_test)
    print("Accuracy for Logistic Regression:\n")
    check_acc(y_test, log_pred)
    #print("Accuracy for Scaled Logistic Regression:\n")
    #check_acc(ys_test, s_log_reg_pred)


def random_forest_classification():
    rfc.fit(X_train, y_train)
    rfc_pred = rfc.predict(X_test)
    print("Accuracy for Random Forest Classification:\n")
    check_acc(y_test, rfc_pred)


def naive_bayes(nb):
    nb.fit(X_train, y_train)
    nb_pred = nb.predict(X_test)
    print("Accuracy for Naive Bayes:\n")
    check_acc(y_test, nb_pred)


def xg_boost():
    xgb.fit(X_train, y_train)
    xgb_pred = xgb.predict(X_test)
    print(xgb_pred)
    print("Accuracy for XGBoost:\n")
    check_acc(y_test, xgb_pred)


def cross_validation_acc(X, est):
    scores = cross_val_score(est, X, y, cv=tscv)
    mean_score = sum(scores) / float(len(scores))
    print("Cross Validation Score:", mean_score, "\n")
    #visualise_cv_split(X)


def visualise_cv_split(data):
    for train_index, test_index in tscv.split(data):
        print("TRAIN:", train_index, "TEST:", test_index)


if __name__ == "__main__":
    df = pd.read_csv('C:/Users/cavan/OneDrive/Documents/PL_ML_Predictions/wpl.csv')#.drop(['Date'], axis=1)  # change path if necessary
    df = df.drop(['Date', 'Season'], axis=1)
    print(df.head)
    X = df.iloc[:, 2:-1].values
    print(X)
    y = df.iloc[:, -1].values
    #df[['homeAF', 'homeDF', 'awayAF', 'awayDF']] = StandardScaler().fit_transform(df[['homeAF', 'homeDF', 'awayAF', 'awayDF']])#, 'LSHAF', 'LSHDF', 'LSAAF', 'LSADF']])
    #X_scaled = df.iloc[:, :-1].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, shuffle=False)
    #Xs_train, Xs_test, ys_train, ys_test = train_test_split(X_scaled, y, test_size=0.1, shuffle=False)
    logistic_regression()
    random_forest_classification()
    naive_bayes(bnb)
    naive_bayes(gnb)
    naive_bayes(mnb)
    naive_bayes(cnb)
    xg_boost()
    print("Unscaled Logistic Regression:\n")
    cross_validation_acc(X, log_reg)
    #print("Scaled Logistic Regression:\n")
    #cross_validation_acc(X_scaled, s_log_reg)
    print("Random Forest Classification:\n")
    cross_validation_acc(X, rfc)
    print("Naive Bayes:\n")
    cross_validation_acc(X, bnb)
    print("XGBoost Algorithm:\n")
    cross_validation_acc(X, xgb)
