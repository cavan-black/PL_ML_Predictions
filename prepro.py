import pandas as pd
from sklearn.preprocessing import LabelEncoder
import glob
import os

le = LabelEncoder()
gamesConsidered = 6 #To Be Optimised!


def merge_csv():
    os.chdir(r"C:\Users\cavan\Downloads\PLRESULTS") #Change File Path if Necessary
    filelist = glob.glob("*.csv")
    dflist = []
    for filename in filelist:
        print(filename)
        df = pd.read_csv(filename, header=None, engine='python')
        dflist.append(df)
    concatdf = pd.concat(dflist, axis=0)
    concatdf.to_csv('C:/Users/cavan/Documents/Diss/PLRES.csv') #Change File Path if Necessary


def homeAttForm(data):
    dfcopy = data.iloc[index:index+gamesConsidered]
    if dfcopy['HomeTeam'].nunique() != 1:
        return 0
    gAve = dfcopy['FTHG'].sum()
    return gAve / gamesConsidered


def homeDefForm(data):
    dfcopy = data.iloc[index:index+gamesConsidered]
    if dfcopy['HomeTeam'].nunique() != 1:
        return 0
    gAve = dfcopy['FTAG'].sum()
    return gAve / gamesConsidered


def awayAttForm(data):
    dfcopy = data.iloc[index:index+gamesConsidered]
    if dfcopy['AwayTeam'].nunique() != 1:
        return 0
    gAve = dfcopy['FTAG'].sum()
    return gAve / gamesConsidered


def awayDefForm(data):
    dfcopy = data.iloc[index:index+gamesConsidered]
    if dfcopy['AwayTeam'].nunique() != 1:
        return 0
    gAve = dfcopy['FTHG'].sum()
    return gAve / gamesConsidered


def do_dummies(data):
    hdummy = pd.get_dummies(data['HomeTeam'])
    adummy = pd.get_dummies(data['AwayTeam'])
    data['HomeTeam'] = le.fit_transform(data['HomeTeam'])
    data['AwayTeam'] = le.transform(data['AwayTeam'])
    data = pd.concat([data,hdummy], axis=1)
    data = pd.concat([data,adummy], axis=1)
    data = data.sort_values(['Date'], ascending=True).reset_index(drop=True)
    data = data.drop(['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG'], axis=1)
    return data


if __name__ == "__main__":
   #merge_csv() #Only call if you want to merge all separate .csv files
    df = pd.read_csv('C:/Users/cavan/Documents/Diss/PLRES.csv', header=0) #Change File Path if Necessary
    df = df.iloc[:, 1:8].reset_index(drop=True).dropna()
    df.columns = ['Div', 'Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR']
    df = df.drop(['Div', 'FTR'], axis =1)
    df['Date'] = pd.to_datetime(df.Date, dayfirst = True)
    df = df.sort_values(['AwayTeam', 'Date'], ascending=[True, False]).reset_index(drop=True)
    for i, (index, row) in enumerate(df.iterrows()):
        if df.at[index, 'FTHG'] > df.at[index, 'FTAG']:
            df.at[index, 'result'] = 1
        elif df.at[index, 'FTHG'] < df.at[index, 'FTAG']:
            df.at[index, 'result'] = 2
        else:
            df.at[index, 'result'] = 0
        df.at[index, 'awayAF'] = (awayAttForm(df))
        df.at[index, 'awayDF'] = (awayDefForm(df))

    df = df.sort_values(['HomeTeam', 'Date'], ascending=[True, False]).reset_index(drop=True)
    for i, (index, row) in enumerate(df.iterrows()):
        df.at[index, 'homeAF'] = (homeAttForm(df))
        df.at[index, 'homeDF'] = (homeDefForm(df))
    df = do_dummies(df)
    cols = list(df.columns.values)
    cols.pop(cols.index('result'))
    df = df[cols + ['result']]
    df.to_csv('C:/Users/cavan/Documents/Diss/wpl.csv') #Change File Path if Necessary







