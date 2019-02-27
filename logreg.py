import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_data(path, header):
    df = pd.read_csv(path, header=header)
    return df


if __name__ == "__main__":
    df = load_data('C:/Users/cavan/Documents/Diss/wpl.csv', 0)

    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    scaler = StandardScaler().fit_transform(x_train)
    log_reg = LogisticRegression(solver='liblinear', multi_class='auto').fit(x_train, y_train)
    acc_train = log_reg.score(x_train, y_train)  # Compute accuracy for training data using Logistic Regression object
    acc_test = log_reg.score(x_test, y_test)  # Compute accuracy for testing data using Logistic Regression object
    print('The accuracy for the training set is:', acc_train)
    print('The accuracy for the testing set is:', acc_test)
