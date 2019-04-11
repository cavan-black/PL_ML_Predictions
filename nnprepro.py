import datetime as datetime
import pandas as pd
import time
from sklearn.preprocessing import LabelEncoder
import os
from datetime import datetime

le = LabelEncoder()
seasonTeams = {
    1: ['Arsenal', 'Aston Villa', 'Blackburn', 'Bolton', 'Chelsea', 'Everton', 'Fulham',
         'Hull', 'Liverpool', 'Man City', 'Man United', 'Middlesbrough', 'Newcastle',
         'Portsmouth', 'Stoke', 'Sunderland', 'Tottenham', 'West Brom', 'West Ham', 'Wigan'],
    2: ['Arsenal', 'Aston Villa', 'Birmingham', 'Blackburn', 'Burnley', 'Bolton', 'Chelsea',
         'Everton', 'Fulham', 'Hull', 'Liverpool', 'Man City', 'Man United',
         'Portsmouth', 'Stoke', 'Sunderland', 'Tottenham', 'West Ham', 'Wigan', 'Wolves'],
    3: ['Arsenal', 'Aston Villa', 'Birmingham', 'Blackburn', 'Blackpool', 'Bolton', 'Chelsea',
         'Everton', 'Fulham', 'Liverpool', 'Man City', 'Man United', 'Newcastle',
         'Stoke', 'Sunderland', 'Tottenham', 'West Brom', 'West Ham', 'Wigan', 'Wolves'],
    4: ['Arsenal', 'Aston Villa', 'Blackburn', 'Bolton', 'Chelsea', 'Everton', 'Fulham',
         'Liverpool', 'Man City', 'Man United', 'Newcastle', 'Norwich', 'QPR',
         'Stoke', 'Sunderland', 'Swansea', 'Tottenham', 'West Brom', 'Wigan', 'Wolves'],
    5: ['Arsenal', 'Aston Villa', 'Chelsea', 'Everton', 'Fulham', 'Liverpool', 'Man City',
        'Man United', 'Newcastle', 'Norwich', 'QPR', 'Reading', 'Southampton', 'Stoke',
         'Sunderland', 'Swansea', 'Tottenham', 'West Brom', 'West Hame', 'Wigan'],
    6: ['Arsenal', 'Aston Villa', 'Cardiff', 'Chelsea', 'Crystal Palace', 'Everton', 'Fulham', 'Hull',
         'Liverpool', 'Man City', 'Man United', 'Newcastle', 'Norwich', 'Southampton', 'Stoke',
         'Sunderland', 'Swansea', 'Tottenham', 'West Brom', 'West Ham'],
    7: ['Arsenal', 'Aston Villa', 'Burnley', 'Chelsea', 'Crystal Palace', 'Everton', 'Hull',
         'Leicester', 'Liverpool', 'Man City', 'Man United', 'Newcastle', 'QPR',
         'Southampton', 'Stoke', 'Sunderland', 'Swansea', 'Tottenham', 'West Brom', 'West Ham'],
    8: ['Arsenal', 'Aston Villa', 'Bournemouth', 'Chelsea', 'Crystal Palace', 'Everton',
         'Leicester', 'Liverpool', 'Man City', 'Man United', 'Newcastle', 'Norwich', 'Southampton',
         'Stoke', 'Sunderland', 'Swansea', 'Tottenham', 'Watford', 'West Brom', 'West Ham'],
    9: ['Arsenal', 'Bournemouth', 'Burnsley', 'Chelsea', 'Crystal Palace', 'Everton', 'Hull',
         'Leicester', 'Liverpool', 'Man City', 'Man United', 'Middlesbrough', 'Southampton',
         'Stoke', 'Sunderland', 'Swansea', 'Tottenham', 'Watford', 'West Brom', 'West Ham'],
    10: ['Arsenal', 'Bournemouth', 'Brighton', 'Burnsley', 'Chelsea', 'Crystal Palace', 'Everton',
         'Huddersfield', 'Leicester', 'Liverpool', 'Man City', 'Man United', 'Newcastle', 'Southampton',
         'Stoke', 'Swansea', 'Tottenham', 'Watford', 'West Brom', 'West Ham'],
    11: ['Arsenal', 'Bournemouth', 'Brighton', 'Burnley', 'Cardiff', 'Chelsea', 'Crystal Palace', 'Everton',
         'Fulham', 'Huddersfield', 'Leicester', 'Liverpool', 'Man City', 'Man United', 'Newcastle',
         'Southampton', 'Tottenham', 'Watford', 'West Ham', 'Wolves']
}
sharedteams = len((set(seasonTeams[1])) & (set(seasonTeams[11])))
dropteams = (sharedteams/2)*(2*(sharedteams-1))
currentTeams = seasonTeams[len(seasonTeams)]
seasonDict = {
    1: {'s': '01/08/2008', 'e': '28/05/2009'}, 2: {'s': '01/08/2009', 'e': '28/05/2010'},
    3: {'s': '01/08/2010', 'e': '28/05/2011'}, 4: {'s': '01/08/2011', 'e': '28/05/2012'},
    5: {'s': '01/08/2012', 'e': '28/05/2013'}, 6: {'s': '01/08/2013', 'e': '28/05/2014'},
    7: {'s': '01/08/2014', 'e': '28/05/2015'}, 8: {'s': '01/08/2015', 'e': '28/05/2016'},
    9: {'s': '01/08/2016', 'e': '28/05/2017'}, 10: {'s': '01/08/2017', 'e': '28/05/2018'},
    11: {'s': '01/08/2018', 'e': '28/05/2019'}
}
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
gamesConsidered = 10

relegations = [relegated_09, relegated_10, relegated_11, relegated_12, relegated_13, relegated_14, relegated_15,
               relegated_16, relegated_17, relegated_18]
promotions = [promoted_09, promoted_10, promoted_11, promoted_12, promoted_13, promoted_14, promoted_15,
              promoted_16, promoted_17, promoted_18]


def pass_on_form2(data):
    data = data.sort_values(['HomeTeam', 'Date'], ascending=[True, True]).reset_index(drop=True)
    print(data)
    for i, (index, row) in enumerate(data.iterrows()):
        season = data.at[index, 'Season']
        if (data.at[index, 'HomeTeam'] in promotions[season-2]) & (season > 1):
            print("Home:")
            print(data.at[index, 'HomeTeam'])
            print(season)
            print(promotions[season-2])
            print(data[index:index+gamesConsidered])
            h_index = promotions[season-2].index(data.at[index, 'HomeTeam'])
            homeSAF, homeSDF, homeSP, homeSC, homeSCR, homeSCFYR = home_season_stats(data, relegations[season-2][h_index],  season - 1)
            data.loc[index, 'HAF'] = homeSAF
            data.loc[index, 'HDF'] = homeSDF
            data.loc[index, 'HPrecision'] = homeSP
            data.loc[index, 'HConversion'] = homeSC
            data.loc[index, 'HConcedeRate'] = homeSCR
            data.loc[index, 'HCFYR'] = homeSCFYR
    data = data.sort_values(['AwayTeam', 'Date'], ascending=[True, True]).reset_index(drop=True)
    for i, (index, row) in enumerate(data.iterrows()):
        season = data.at[index, 'Season']
        if (data.at[index, 'AwayTeam'] in promotions[season-2]) & (season > 1):
            a_index = promotions[season - 2].index(data.at[index, 'AwayTeam'])
            print("Away:")
            print(season)
            print(promotions[season - 2])
            awaySAF, awaySDF, awaySP, awaySC, awaySCR, awaySCFYR = away_season_stats(data, relegations[season-2][a_index],  season - 1)
            data.at[index, 'AAF'] = awaySAF
            data.at[index, 'ADF'] = awaySDF
            data.at[index, 'APrecision'] = awaySP
            data.at[index, 'AConversion'] = awaySC
            data.at[index, 'AConcedeRate'] = awaySCR
            data.at[index, 'ACFYR'] = awaySCFYR
    return data

def pass_on_form(data):
    for count, releg in enumerate(relegations):
        for index, team in enumerate(releg):
            print(index)
            print(count)
            print(team, "==>", promotions[count][index])
            homeSAF, homeSDF, homeSP, homeSC, homeSCR, homeSCFYR = home_season_stats(data, team, count+1)
            #print(homeSAF, homeSDF, homeSP, homeSC, homeSCR, homeSCFYR)
            awaySAF, awaySDF, awaySP, awaySC, awaySCR, awaySCFYR = away_season_stats(data, team, count+1)
            #print(awaySAF, awaySDF, awaySP, awaySC, awaySCR, awaySCFYR)
            pass_on_hf(data, promotions[count][index], count+2, homeSAF, homeSDF, homeSP, homeSC, homeSCR, homeSCFYR)
            pass_on_af(data, promotions[count][index], count+2, awaySAF, awaySDF, awaySP, awaySC, awaySCR, awaySCFYR)
        #print("Pass On Form:", count / len(relegations) * 100, "%")
    return data


def pass_on_hf(data, team, season, seasonAF, seasonDF, seasonP, seasonC, seasonCR, seasonCFYR):
    data = data.loc[data['HomeTeam'] == team]
    data = slice_dates(data, season)
    data = data.iloc[0:gamesConsidered-1]
    for i, (index, row) in enumerate(data.iterrows()):
        data.loc[index, 'HAF'] = seasonAF
        data.loc[index, 'HDF'] = seasonDF
        data.loc[index, 'HPrecision'] = seasonP
        data.loc[index, 'HConversion'] = seasonC
        data.loc[index, 'HConcedeRate'] = seasonCR
        data.loc[index, 'HCFYR'] = seasonCFYR
    return data


def pass_on_af(data, team, season, seasonAF, seasonDF, seasonP, seasonC, seasonCR, seasonCFYR):
    data = data.loc[data['AwayTeam'] == team]
    data = slice_dates(data, season)
    data = data.iloc[0:gamesConsidered-1]
    print(data)
    for i, (index, row) in enumerate(data.iterrows()):
        data.at[index, 'AAF'] = seasonAF
        data.at[index, 'ADF'] = seasonDF
        data.at[index, 'APrecision'] = seasonP
        print(data.at[index, 'APrecision'])
        data.at[index, 'AConversion'] = seasonC
        data.at[index, 'AConcedeRate'] = seasonCR
        data.at[index, 'ACFYR'] = seasonCFYR
        print("After:", data)
    return data


def home_season_stats(data, homeTeam, season):
    if season == 0:
        return 0
    dfcopy = data.copy()
    dfcopy = dfcopy.loc[dfcopy['HomeTeam'] == homeTeam]
    dfcopy = slice_dates(dfcopy, season)
    if len(dfcopy.index) == 0:
        return 0
    homeSAF = dfcopy['HAF'].sum()/len(dfcopy.index)
    homeSDF = dfcopy['HDF'].sum()/len(dfcopy.index)
    homeSP = dfcopy['HPrecision'].sum()/len(dfcopy.index)
    homeSC = dfcopy['HConversion'].sum()/len(dfcopy.index)
    homeSCR = dfcopy['HConcedeRate'].sum()/len(dfcopy.index)
    homeSCFYR = dfcopy['HCFYR'].sum()/len(dfcopy.index)
    return homeSAF, homeSDF, homeSP, homeSC, homeSCR, homeSCFYR


def away_season_stats(data, awayTeam, season):
    if season == 0:
        return 0
    dfcopy = data.copy()
    dfcopy = dfcopy.loc[dfcopy['AwayTeam'] == awayTeam]
    dfcopy = slice_dates(dfcopy, season)
    if len(dfcopy.index) == 0:
        return 0
    awaySAF = dfcopy['AAF'].sum()/len(dfcopy.index)
    awaySDF = dfcopy['ADF'].sum()/len(dfcopy.index)
    awaySP = dfcopy['APrecision'].sum()/len(dfcopy.index)
    awaySC = dfcopy['AConversion'].sum()/len(dfcopy.index)
    awaySCR = dfcopy['AConcedeRate'].sum()/len(dfcopy.index)
    awaySCFYR = dfcopy['ACFYR'].sum()/len(dfcopy.index)
    #print("Away team:", awayTeam, "\nStats:", awaySAF, awaySDF, awaySP, awaySC, awaySCR, awaySCFYR)
    return awaySAF, awaySDF, awaySP, awaySC, awaySCR, awaySCFYR


def slice_dates(data, season):
    seasonStart = seasonDict[season]['s']
    seasonEnd = seasonDict[season]['e']
    #print(seasonStart, seasonEnd)
    data['Date'] = pd.to_datetime(data.Date, dayfirst=True)
    data = data.set_index(['Date'], drop=True).sort_values(['Date'], ascending=True)
    data = data.loc[seasonStart:seasonEnd]
    #print(data.head)
    return data


def h_average_vals(data, index, colname, gamesConsidered):
    dfcopy = data.iloc[index + 1:index + gamesConsidered + 1]
    dfcheck = data.iloc[index:index + gamesConsidered]
    if (dfcheck['HomeTeam'].nunique() != 1) | (len(dfcopy.index) < gamesConsidered):
        return 0
    total = dfcopy[colname].sum()
    return total


def a_average_vals(data, index, colname, gamesConsidered):
    dfcopy = data.iloc[index + 1:index + gamesConsidered + 1]
    dfcheck = data.iloc[index:index + gamesConsidered]
    if (dfcheck['AwayTeam'].nunique() != 1) | (len(dfcopy.index) < gamesConsidered):
        return 0
    total = dfcopy[colname].sum()
    return total


def calc_cols(data):
    data = data.sort_values(['HomeTeam', 'Date'], ascending=[True, False]).reset_index(drop=True)
    for i, (index, row) in enumerate(data.iterrows()):
        data.at[index, 'HAF'] = (h_average_vals(data, index, 'FTHG', gamesConsidered=gamesConsidered))/gamesConsidered
        data.at[index, 'HDF'] = (h_average_vals(data, index, 'FTAG', gamesConsidered=gamesConsidered))/gamesConsidered
        homeShots = h_average_vals(data, index, 'HS', gamesConsidered)
        awayShots = h_average_vals(data, index, 'AS', gamesConsidered)
        if homeShots == 0:
            homeShots = 1
        if awayShots == 0:
            awayShots = 1
        data.at[index, 'HPrecision'] = (h_average_vals(data, index, 'HST', gamesConsidered)/
                                        homeShots)
        data.at[index, 'HConversion'] = (h_average_vals(data, index, 'FTHG', gamesConsidered)/
                                         homeShots)
        data.at[index, 'HConcedeRate'] = (h_average_vals(data, index, 'FTAG', gamesConsidered)/
                                          awayShots)
        data.at[index, 'HCFYR'] = ((h_average_vals(data, index, 'HF', gamesConsidered)+
                                   h_average_vals(data, index, 'AC', gamesConsidered)+
                                   h_average_vals(data, index, 'HY', gamesConsidered)+
                                   h_average_vals(data, index, 'HR', gamesConsidered))/gamesConsidered)
        print("Calculate Form:", i/len(data)*50, "%")

    data = data.sort_values(['AwayTeam', 'Date'], ascending=[True, False]).reset_index(drop=True)
    print(data.head)
    for i, (index, row) in enumerate(data.iterrows()):
        awayShots = a_average_vals(data, index, 'AS', gamesConsidered)
        homeShots = a_average_vals(data, index, 'HS', gamesConsidered)
        if awayShots == 0:
            awayShots = 1
        if homeShots == 0:
            homeShots = 1
        data.at[index, 'AAF'] = (a_average_vals(data, index, 'FTAG', gamesConsidered=gamesConsidered)) / gamesConsidered
        data.at[index, 'ADF'] = (a_average_vals(data, index, 'FTHG', gamesConsidered=gamesConsidered)) / gamesConsidered
        data.at[index, 'APrecision'] = (a_average_vals(data, index, 'AST', gamesConsidered) /
                                        awayShots)
        data.at[index, 'AConversion'] = (a_average_vals(data, index, 'FTAG', gamesConsidered) /
                                         awayShots)
        data.at[index, 'AConcedeRate'] = (a_average_vals(data, index, 'FTHG', gamesConsidered) /
                                          homeShots)
        data.at[index, 'ACFYR'] = ((a_average_vals(data, index, 'AF', gamesConsidered) +
                                    a_average_vals(data, index, 'HC', gamesConsidered) +
                                    a_average_vals(data, index, 'AY', gamesConsidered) +
                                    a_average_vals(data, index, 'AR', gamesConsidered)) / gamesConsidered)
        print("Calculate Form:", (i/len(data)*50)+50, "%")
    data = data.drop(['FTHG', 'FTAG', 'HS', 'AS', 'HST', 'AST', 'HF', 'AF', 'HC', 'AC', 'HY', 'AY', 'HR', 'AR'], axis=1)
    data = data.sort_values(['Date'], ascending=True).reset_index(drop=True)
    data.to_csv('skips.csv', index =False)
    return data


def refine_columns(data):
    data = data.drop(['HTHG', 'HTAG', 'HTR', 'Referee'], axis=1)
    data['Date'] = pd.to_datetime(data.Date, dayfirst=True)
    data = data.sort_values(['HomeTeam', 'Date'], ascending=[True, False]).reset_index(drop=True)
    return data


def merge_csv():
    os.chdir(r"C:/Users/cavan/Downloads/NNPL")  # Change File Path if Necessary
    dflist = []
    for file in range(1, 12):
        filename = str(file)+".csv"
        df1 = pd.read_csv(filename, header=0, engine='python', usecols=(1, 2, 3, 4, 5, 6, 11, 12 ,13, 14, 15, 16, 17,
                                                                        18, 19, 20, 21, 22)).dropna()
        df1['Season'] = file
        dflist.append(df1)
    concatdf = pd.concat(dflist, axis=0, sort=False)
    concatdf.to_csv('C:/Users/cavan/OneDrive/Documents/PL_ML_Predictions/NNPLR.csv')  # Change File Path if Necessary


def remove_teams(data):
    data = data.loc[data['HomeTeam'].isin(currentTeams)]
    data = data.loc[data['AwayTeam'].isin(currentTeams)]
    return data


def do_dummies(data):
    hdummy = pd.get_dummies('H_' + data['HomeTeam'])
    adummy = pd.get_dummies('A_' + data['AwayTeam'])
    data['HomeTeam'] = le.fit_transform(data['HomeTeam'])
    data['AwayTeam'] = le.transform(data['AwayTeam'])
    data = pd.concat([data, hdummy], axis=1)
    data = pd.concat([data, adummy], axis=1)
    data = data.sort_index(ascending=True)
    data = data.drop(['HomeTeam', 'AwayTeam'], axis=1).reset_index(drop=True)
    return data


def reorganise_columns(data):
    moveres = data.pop('FTR')  # Remove result column and store it in df1
    data['FTR'] = moveres
    data = data.reset_index(drop=True)
    data = data.drop(data.index[0:dropteams])#.reset_index(drop=True)
    return data


def convert_dict():
    for item in seasonDict:
        seasonDict[item]['s'] = datetime.strptime(seasonDict[item]['s'], '%d/%m/%Y')
        seasonDict[item]['e'] = datetime.strptime(seasonDict[item]['e'], '%d/%m/%Y')


def numres(data):
    for i, (index,row) in enumerate(data.iterrows()):
        if data.at[index, 'FTR'] == 'H':
            data.at[index, 'FTR'] = '1'  # 1 means a home win
        elif data.at[index, 'FTR'] == 'A':
            data.at[index, 'FTR'] = '2'  # 2 means an away win
        else:
            data.at[index, 'FTR'] = '0'  # 0 means a draw
    return data

if __name__ == "__main__":
    #merge_csv()  # Only call if you want to merge all separate .csv files
    #os.chdir('C:/Users/cavan/OneDrive/Documents/PL_ML_Predictions')# Change File Path if Necessary
    #df = pd.read_csv('NNPLR.csv', header=0).drop(['Unnamed: 0'], axis=1)
    #df['Date'] = pd.to_datetime(df.Date, dayfirst=True)
    #df = calc_cols(df)
    #convert_dict()
    df = pd.read_csv('skips.csv')

    df = pass_on_form2(df).sort_values(['Date'], ascending=True)
    df.to_csv('Checkup.csv')
    df = remove_teams(df)
    #df2 = reorganise_columns(df)
    #df2.to_csv('nodums.csv', index=False)
    df = do_dummies(df)
    df = reorganise_columns(df)
    df.to_csv('letres.csv')
    df = numres(df)
    df.to_csv('wpl.csv', index=False)  # Change File Path if Necessary
    #print(df.columns.values)
