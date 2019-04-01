import pandas as pd
from sklearn.preprocessing import LabelEncoder
import glob
import os

le = LabelEncoder()
gamesConsidered = 5  # To Be Optimised!
currentTeams = ['Arsenal', 'Bournemouth', 'Brighton', 'Burnley', 'Cardiff', 'Chelsea', 'Crystal Palace', 'Everton',
                'Fulham', 'Huddersfield', 'Leicester', 'Liverpool', 'Man City', 'Man United', 'Newcastle', 'Southampton'
                , 'Tottenham', 'Watford', 'West Ham', 'Wolves']
seasonDict = {
    1: {'s': '08/2004', 'e': '06/2005'},
    2: {'s': '08/2005', 'e': '06/2006'},
    3: {'s': '08/2006', 'e': '06/2007'},
    4: {'s': '08/2007', 'e': '06/2008'},
    5: {'s': '08/2008', 'e': '06/2009'},
    6: {'s': '08/2009', 'e': '06/2010'},
    7: {'s': '08/2010', 'e': '06/2011'},
    8: {'s': '08/2011', 'e': '06/2012'},
    9: {'s': '08/2012', 'e': '06/2013'},
    10: {'s': '08/2013', 'e': '06/2014'},
    11: {'s': '08/2014', 'e': '06/2015'},
    12: {'s': '08/2015', 'e': '06/2016'},
    13: {'s': '08/2016', 'e': '06/2017'},
    14: {'s': '08/2017', 'e': '06/2018'},
    15: {'s': '08/2018', 'e': '06/2019'}
}
relegated_05 = ["Crystal Palace", "Norwich", "Southampton"]
promoted_05 = ["Reading", "Sheffield", "Watford"]
relegated_06 = ["Birmingham", "West Brom", "Sunderland"]
promoted_06 = ["Reading", "Sheffield", "Watford"]
relegated_07 = ["Sheffield", "Charlton", "Watford"]
promoted_07 = ["Sunderland", "Birmingham", "Derby"]
relegated_08 = ["Reading", "Birmingham", "Derby"]
promoted_08 = ["West Brom", "Stoke", "Hull"]
relegated_09 = ["Newcastle", "Middlesbrough", "West Brom"]
promoted_09 = ["Wolves", "Birmingham", "Burnley"]
relegated_10 = ["Burnley", "Hull", "Portsmouth"]
promoted_10 = ["Newcastle", "West Brom", "Blackpool"]
relegated_11 = ["Birmingham", "Blackpool", "West Ham"]
promoted_11 = ["QPR", "Norwich", "Swansea"]
relegated_12 = ["Bolton", "Blackburn", "Wolves"]
promoted_12 = ["Reading", "Southampton", "West Ham"]
relegated_13 = ["Wigan", "Reading", "QPR"]
promoted_13 = ["Cardiff", "Hull", "Crystal Palace"]
relegated_14 = ["Norwich", "Fulham", "Cardiff"]
promoted_14 = ["Leicester", "Burnley", "QPR"]
relegated_15 = ["Hull", "Burnley", "QPR"]
promoted_15 = ["Bournemouth", "Watford", "Norwich"]
relegated_16 = ["Newcastle", "Norwich", "Aston Villa"]
promoted_16 = ["Burnley", "Middlesbrough", "Hull"]
relegated_17 = ["Hull", "Middlesbrough", "Sunderland"]
promoted_17 = ["Newcastle", "Brighton", "Huddersfield"]
relegated_18 = ["Swansea", "Stoke", "West Brom"]
promoted_18 = ["Wolves", "Cardiff", "Fulham"]

relegations = [relegated_05, relegated_06, relegated_07, relegated_08, relegated_09, relegated_10, relegated_11,
               relegated_12, relegated_13, relegated_14, relegated_15, relegated_16, relegated_17, relegated_18]
promotions = [promoted_05, promoted_06, promoted_07, promoted_08, promoted_09, promoted_10, promoted_11, promoted_12,
              promoted_13, promoted_14, promoted_15, promoted_16, promoted_17, promoted_18]


def pass_on_form(data):
    for count, releg in enumerate(relegations):
        print(count)
        for index, team in enumerate(releg):
            print(team)
            homeSAF, homeSDF = home_season_form(data, team, count+1)
            awaySAF, awaySDF = away_season_form(data, team, count+1)
            pass_on_hf(data, promotions[count][index], count+2, homeSAF, homeSDF)
            pass_on_af(data, promotions[count][index], count+2, awaySAF, awaySDF)
    return data


def pass_on_hf(data, team, season, seasonAF, seasonDF):
    data = data.loc[data['HomeTeam'] == team]
    data = slice_dates(data, season)
    data = data.iloc[0:gamesConsidered]
    print(seasonAF, seasonDF)
    for i, (index, row) in enumerate(data.iterrows()):
        data.at[index, 'homeAF'] = seasonAF
        data.at[index, 'homeDF'] = seasonDF
    return data


def pass_on_af(data, team, season, seasonAF, seasonDF):
    data = data.loc[data['AwayTeam'] == team]
    data = slice_dates(data, season)
    data = data.iloc[0:gamesConsidered]
    for i, (index, row) in enumerate(data.iterrows()):
        data.at[index, 'awayAF'] = seasonAF
        data.at[index, 'awayDF'] = seasonDF
    return data


def slice_dates(data, season):
    seasonStart = seasonDict[season]['s']
    seasonEnd = seasonDict[season]['e']
    data = data.set_index(['Date']).sort_values(['Date'], ascending=True)
    data = data.loc[seasonStart:seasonEnd]
    return data


def home_season_form(data, homeTeam, season):
    dfcopy = data.copy()
    dfcopy = dfcopy.loc[dfcopy['HomeTeam'] == homeTeam]
    dfcopy = slice_dates(dfcopy, season)
    homeSAF = dfcopy['homeAF'].sum()/19
    homeSDF = dfcopy['homeDF'].sum()/19
    return homeSAF, homeSDF


def away_season_form(data, awayTeam, season):
    dfcopy = data.copy()
    dfcopy = dfcopy.loc[dfcopy['AwayTeam'] == awayTeam]
    dfcopy = slice_dates(dfcopy, season)
    awaySAF = dfcopy['awayAF'].sum()/19
    awaySDF = dfcopy['awayDF'].sum()/19
    return awaySAF, awaySDF


def merge_csv():
    os.chdir(r"C:/Users/cavan/OneDrive/Downloads/PLRESULTS")  # Change File Path if Necessary
    filelist = glob.glob("*.csv")
    dflist = []
    for filename in filelist:
        print(filename)
        df1 = pd.read_csv(filename, header=0, engine='python')
        df1 = df1.drop(df1.index[0], inplace=True)
        dflist.append(df1)
    concatdf = pd.concat(dflist, axis=0)
    concatdf.to_csv('C:/Users/cavan/OneDrive/Documents/PL_ML_Predictions/PLRES.csv')  # Change File Path if Necessary


def home_att_form(data, index):
    dfcopy = data.iloc[index:index+gamesConsidered]
    if dfcopy['HomeTeam'].nunique() != 1:
        return 0
    gAve = dfcopy['FTHG'].sum()
    return gAve / gamesConsidered


def home_def_form(data, index):
    dfcopy = data.iloc[index:index+gamesConsidered]
    if dfcopy['HomeTeam'].nunique() != 1:
        return 0
    gAve = dfcopy['FTAG'].sum()
    return gAve / gamesConsidered


def away_att_form(data, index):
    dfcopy = data.iloc[index:index+gamesConsidered]
    if dfcopy['AwayTeam'].nunique() != 1:
        return 0
    gAve = dfcopy['FTAG'].sum()
    return gAve / gamesConsidered


def away_def_form(data, index):
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
    data = pd.concat([data, hdummy], axis=1)
    data = pd.concat([data, adummy], axis=1)
    data = data.sort_values(['Date'], ascending=True).reset_index(drop=True)
    data = data.drop(['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG'], axis=1)
    return data


def calc_form(data):
    for i, (index, row) in enumerate(data.iterrows()):
        if data.at[index, 'FTHG'] > data.at[index, 'FTAG']:
            data.at[index, 'result'] = 1  # 1 means a home win
        elif data.at[index, 'FTHG'] < data.at[index, 'FTAG']:
            data.at[index, 'result'] = 2  # 2 means an away win
        else:
            data.at[index, 'result'] = 0  # 0 means a draw
        data.at[index, 'awayAF'] = (away_att_form(data, index))
        data.at[index, 'awayDF'] = (away_def_form(data, index))
    data = data.sort_values(['HomeTeam', 'Date'], ascending=[True, False]).reset_index(drop=True)
    for i, (index, row) in enumerate(data.iterrows()):
        data.at[index, 'homeAF'] = (home_att_form(data, index))
        data.at[index, 'homeDF'] = (home_def_form(data, index))
    return data


def reorganise_columns(data):
    moveres = data.pop('result')  # Remove result column and store it in df1
    data['result'] = moveres
    data = data.drop(data.index[0:((gamesConsidered - 1) * 20) - 1])  # .reset_index(drop=True)
    return data


def refine_columns(data):
    data = data.iloc[:, 1:8].reset_index(drop=True).dropna()
    data.columns = ['Div', 'Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR']
    data = data.drop(['Div', 'FTR'], axis=1)
    data['Date'] = pd.to_datetime(data.Date, dayfirst=True)
    data = data.sort_values(['AwayTeam', 'Date'], ascending=[True, False]).reset_index(drop=True)
    return data


def remove_teams(data):
    data = data.loc[data['HomeTeam'].isin(currentTeams)]
    data = data.loc[data['AwayTeam'].isin(currentTeams)]
    data.to_csv('C:/Users/cavan/OneDrive/Documents/PL_ML_Predictions/wsxspl.csv')
    return data


if __name__ == "__main__":
    # merge_csv()  # Only call if you want to merge all separate .csv files
    df = pd.read_csv('C:/Users/cavan/OneDrive/Documents/PL_ML_Predictions/PLRES.csv', header=0)
    # Change File Path if Necessary
    df = refine_columns(df)
    df = calc_form(df)
    df = pass_on_form(df)
    df = remove_teams(df)
    df = do_dummies(df)
    df = reorganise_columns(df)
    df.to_csv('C:/Users/cavan/OneDrive/Documents/PL_ML_Predictions/wpl.csv')  # Change File Path if Necessary
