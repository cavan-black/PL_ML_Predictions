import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import BernoulliNB


def plot_conf_matrix(conf_matrix):
    plt.figure(figsize=(5, 5))
    sns.heatmap(conf_matrix, annot=True, cmap="YlGnBu", fmt='g')
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')
    plt.show()


def check_acc(y_test, pred):
    print("The Accuracy using accuracy_score is", accuracy_score(y_test, pred))
    conf_matrix = confusion_matrix(y_test, pred)
    plot_conf_matrix(conf_matrix)
    print('Classification Report: \n\n', classification_report(y_test, pred))  # , target_names = seed_names))


def random_forest_classification(X_train, X_test, y_train, y_test):
    rfc = RandomForestClassifier(n_estimators=100).fit(X_train, y_train)
    rfc_pred = rfc.predict(X_test)
    print("Accuracy for Random Forest Classification:\n")
    check_acc(y_test, rfc_pred)


def logistic_regression(X_train, X_test, y_train, y_test):
    StandardScaler().fit_transform(X_train)
    log_reg = LogisticRegression(solver='liblinear', multi_class='auto').fit(X_train, y_train)
    log_pred = log_reg.predict(X_test)
    print("Accuracy for Logistic Regression:\n")
    check_acc(y_test, log_pred)


def naive_bayes(X_train, X_test, y_train, y_test):
    nb = BernoulliNB().fit(X_train, y_train)
    nbpred = nb.predict(X_test)
    print("Accuracy for Naive Bayes:\n")
    check_acc(y_test, nbpred)


def split_data(df):
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    df = pd.read_csv('C:/Users/cavan/OneDrive/Documents/PL_ML_Predictions/wpl.csv')  # change path if necessary
    X_train, X_test, y_train, y_test = split_data(df)
    logistic_regression(X_train, X_test, y_train, y_test)
    random_forest_classification(X_train, X_test, y_train, y_test)
    naive_bayes(X_train, X_test, y_train, y_test)





