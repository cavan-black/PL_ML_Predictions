import datetime as datetime
import pandas as pd
import time
from sklearn.preprocessing import LabelEncoder
import os
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from csv import DictReader


le = LabelEncoder()
gamesConsidered = 10 # To Be Optimised!
seasonDict = {
    1: {'s': '01/08/1994', 'e': '28/05/1995'},
    2: {'s': '01/08/1995', 'e': '28/05/1996'},
    3: {'s': '01/08/1996', 'e': '28/05/1997'},
    4: {'s': '01/08/1997', 'e': '28/05/1998'},
    5: {'s': '01/08/1998', 'e': '28/05/1999'},
    6: {'s': '01/08/1999', 'e': '28/05/2000'},
    7: {'s': '01/08/2000', 'e': '28/05/2001'},
    8: {'s': '01/08/2001', 'e': '28/05/2002'},
    9: {'s': '01/08/2002', 'e': '28/05/2003'},
    10: {'s': '01/08/2003', 'e': '28/05/2004'},
    11: {'s': '01/08/2004', 'e': '28/05/2005'},
    12: {'s': '01/08/2005', 'e': '28/05/2006'},
    13: {'s': '01/08/2006', 'e': '28/05/2007'},
    14: {'s': '01/08/2007', 'e': '28/05/2008'},
    15: {'s': '01/08/2008', 'e': '28/05/2009'},
    16: {'s': '01/08/2009', 'e': '28/05/2010'},
    17: {'s': '01/08/2010', 'e': '28/05/2011'},
    18: {'s': '01/08/2011', 'e': '28/05/2012'},
    19: {'s': '01/08/2012', 'e': '28/05/2013'},
    20: {'s': '01/08/2013', 'e': '28/05/2014'},
    21: {'s': '01/08/2014', 'e': '28/05/2015'},
    22: {'s': '01/08/2015', 'e': '28/05/2016'},
    23: {'s': '01/08/2016', 'e': '28/05/2017'},
    24: {'s': '01/08/2017', 'e': '28/05/2018'},
    25: {'s': '01/08/2018', 'e': '28/05/2019'}
}
seasonTeams = {
    1: ['Arsenal', 'Aston Villa', 'Blackburn', 'Chelsea', 'Coventry', 'Crystal Palace', 'Everton', 'Ipswich', 'Leeds',
        'Leicester', 'Liverpool', 'Man City', 'Man United', 'Newcastle', 'Norwich', "Notts Forest", 'QPR',
        'Sheffield Weds', 'Southampton', 'Tottenham', 'West Ham', 'Wimbledon'],
    2: ['Arsenal', 'Aston Villa', 'Blackburn', 'Bolton', 'Chelsea', 'Coventry', 'Everton', 'Leeds',
        'Liverpool', 'Man City', 'Man United', 'Middlesbrough', 'Newcastle', "Notts Forest", 'QPR',
        'Sheffield Weds', 'Southampton', 'Tottenham', 'West Ham', 'Wimbledon'],
    3: ['Arsenal', 'Aston Villa', 'Blackburn', 'Chelsea', 'Coventry', 'Derby', 'Everton', 'Leeds', 'Leicester',
        'Liverpool', 'Man United', 'Middlesbrough', 'Newcastle', "Notts Forest",
        'Sheffield Weds', 'Southampton', 'Sunderland', 'Tottenham', 'West Ham', 'Wimbledon'],
    4: ['Arsenal', 'Aston Villa', 'Barnsley', 'Blackburn', 'Bolton', 'Chelsea', 'Coventry', 'Crystal Palace',
        'Derby', 'Everton', 'Leeds', 'Leicester', 'Liverpool', 'Man United', 'Newcastle',
        'Sheffield Weds', 'Southampton', 'Tottenham', 'West Ham', 'Wimbledon'],
    5: ['Arsenal', 'Aston Villa', 'Blackburn', 'Charlton', 'Chelsea', 'Coventry',
        'Derby', 'Everton', 'Leeds', 'Leicester', 'Liverpool', 'Man United', 'Middlesbrough', 'Newcastle', "Notts Forest",
        'Sheffield Weds', 'Southampton', 'Tottenham', 'West Ham', 'Wimbledon'],
    6: ['Arsenal', 'Aston Villa', 'Bradford', 'Chelsea', 'Coventry', 'Derby',
        'Everton', 'Leeds', 'Leicester', 'Liverpool', 'Man United', 'Middlesbrough', 'Newcastle',
        'Sheffield Weds', 'Southampton', 'Sunderland', 'Tottenham', 'Watford', 'West Ham', 'Wimbledon'],
    7: ['Arsenal', 'Aston Villa', 'Bradford', 'Charlton', 'Chelsea', 'Coventry', 'Derby',
        'Everton', 'Ipswich', 'Leeds', 'Leicester', 'Liverpool', 'Man City', 'Man United', 'Middlesbrough', 'Newcastle',
        'Southampton', 'Sunderland', 'Tottenham', 'West Ham'],
    8: ['Arsenal', 'Aston Villa', 'Blackburn', 'Bolton', 'Charlton', 'Chelsea', 'Derby', 'Everton',
        'Fulham', 'Ipswich', 'Leeds', 'Leicester', 'Liverpool', 'Man United', 'Middlesbrough',
        'Newcastle', 'Southampton', 'Sunderland', 'Tottenham', 'West Ham'],
    9: ['Arsenal', 'Aston Villa', 'Birmingham', 'Blackburn', 'Bolton', 'Charlton', 'Chelsea', 'Everton',
        'Fulham', 'Leeds', 'Liverpool', 'Man City', 'Man United', 'Middlesbrough',
        'Newcastle', 'Southampton', 'Sunderland', 'Tottenham', 'West Brom', 'West Ham'],
    10: ['Arsenal', 'Aston Villa', 'Birmingham', 'Blackburn', 'Bolton', 'Charlton', 'Chelsea', 'Everton',
        'Fulham', 'Leeds', 'Leicester', 'Liverpool', 'Man City', 'Man United', 'Middlesbrough',
        'Newcastle', 'Portsmouth', 'Southampton', 'Tottenham', 'Wolves'],
    11: ['Arsenal', 'Aston Villa', 'Birmingham', 'Blackburn', 'Bolton', 'Charlton', 'Chelsea', 'Crystal Palace',
         'Everton', 'Fulham', 'Liverpool', 'Man City', 'Man United', 'Middlesbrough',
        'Newcastle', 'Norwich', 'Portsmouth', 'Southampton', 'Tottenham', 'West Brom'],
    12: ['Arsenal', 'Aston Villa', 'Birmingham', 'Blackburn', 'Bolton', 'Charlton', 'Chelsea',
        'Everton', 'Fulham', 'Liverpool', 'Man City', 'Man United', 'Middlesbrough',
        'Newcastle', 'Portsmouth', 'Sunderland', 'Tottenham', 'West Brom', 'West Ham', 'Wigan'],
    13: ['Arsenal', 'Aston Villa', 'Blackburn', 'Bolton', 'Charlton', 'Chelsea',
        'Everton', 'Fulham', 'Liverpool', 'Man City', 'Man United', 'Middlesbrough', 'Newcastle',
         'Portsmouth', 'Reading', 'Sheffield United', 'Tottenham', 'Watford', 'West Ham', 'Wigan'],
    14: ['Arsenal', 'Aston Villa', 'Birmingham', 'Blackburn', 'Bolton', 'Chelsea', 'Derby',
        'Everton', 'Fulham', 'Liverpool', 'Man City', 'Man United', 'Middlesbrough', 'Newcastle',
         'Portsmouth', 'Reading', 'Sunderland', 'Tottenham', 'West Ham', 'Wigan'],
    15: ['Arsenal', 'Aston Villa', 'Blackburn', 'Bolton', 'Chelsea', 'Everton', 'Fulham',
         'Hull', 'Liverpool', 'Man City', 'Man United', 'Middlesbrough', 'Newcastle',
         'Portsmouth', 'Stoke', 'Sunderland', 'Tottenham', 'West Brom', 'West Ham', 'Wigan'],
    16: ['Arsenal', 'Aston Villa', 'Birmingham', 'Blackburn', 'Burnley', 'Bolton', 'Chelsea',
         'Everton', 'Fulham', 'Hull', 'Liverpool', 'Man City', 'Man United',
         'Portsmouth', 'Stoke', 'Sunderland', 'Tottenham', 'West Ham', 'Wigan', 'Wolves'],
    17: ['Arsenal', 'Aston Villa', 'Birmingham', 'Blackburn', 'Blackpool', 'Bolton', 'Chelsea',
         'Everton', 'Fulham', 'Liverpool', 'Man City', 'Man United', 'Newcastle',
         'Stoke', 'Sunderland', 'Tottenham', 'West Brom', 'West Ham', 'Wigan', 'Wolves'],
    18: ['Arsenal', 'Aston Villa', 'Blackburn', 'Bolton', 'Chelsea', 'Everton', 'Fulham',
         'Liverpool', 'Man City', 'Man United', 'Newcastle', 'Norwich', 'QPR',
         'Stoke', 'Sunderland', 'Swansea', 'Tottenham', 'West Brom', 'Wigan', 'Wolves'],
    19: ['Arsenal', 'Aston Villa', 'Chelsea', 'Everton', 'Fulham', 'Liverpool', 'Man City',
         'Man United', 'Newcastle', 'Norwich', 'QPR', 'Reading', 'Southampton', 'Stoke',
         'Sunderland', 'Swansea', 'Tottenham', 'West Brom', 'West Hame', 'Wigan'],
    20: ['Arsenal', 'Aston Villa', 'Cardiff', 'Chelsea', 'Crystal Palace', 'Everton', 'Fulham', 'Hull',
         'Liverpool', 'Man City', 'Man United', 'Newcastle', 'Norwich', 'Southampton', 'Stoke',
         'Sunderland', 'Swansea', 'Tottenham', 'West Brom', 'West Ham'],
    21: ['Arsenal', 'Aston Villa', 'Burnley', 'Chelsea', 'Crystal Palace', 'Everton', 'Hull',
         'Leicester', 'Liverpool', 'Man City', 'Man United', 'Newcastle', 'QPR',
         'Southampton', 'Stoke', 'Sunderland', 'Swansea', 'Tottenham', 'West Brom', 'West Ham'],
    22: ['Arsenal', 'Aston Villa', 'Bournemouth', 'Chelsea', 'Crystal Palace', 'Everton',
         'Leicester', 'Liverpool', 'Man City', 'Man United', 'Newcastle', 'Norwich', 'Southampton',
         'Stoke', 'Sunderland', 'Swansea', 'Tottenham', 'Watford', 'West Brom', 'West Ham'],
    23: ['Arsenal', 'Bournemouth', 'Burnsley', 'Chelsea', 'Crystal Palace', 'Everton', 'Hull',
         'Leicester', 'Liverpool', 'Man City', 'Man United', 'Middlesbrough', 'Southampton',
         'Stoke', 'Sunderland', 'Swansea', 'Tottenham', 'Watford', 'West Brom', 'West Ham'],
    24: ['Arsenal', 'Bournemouth', 'Brighton', 'Burnsley', 'Chelsea', 'Crystal Palace', 'Everton',
         'Huddersfield', 'Leicester', 'Liverpool', 'Man City', 'Man United', 'Newcastle', 'Southampton',
         'Stoke', 'Swansea', 'Tottenham', 'Watford', 'West Brom', 'West Ham'],
    25: ['Arsenal', 'Bournemouth', 'Brighton', 'Burnley', 'Cardiff', 'Chelsea', 'Crystal Palace', 'Everton',
         'Fulham', 'Huddersfield', 'Leicester', 'Liverpool', 'Man City', 'Man United', 'Newcastle',
         'Southampton', 'Tottenham', 'Watford', 'West Ham', 'Wolves']
}
sharedteams = len((set(seasonTeams[1])) & (set(seasonTeams[25])))
dropteams = (sharedteams/2)*(2*(sharedteams-1))
currentTeams = seasonTeams[len(seasonTeams)]
relegated_95 = ["Crystal Palace", "Norwich"]
promoted_95 = ["Middlesbrough", "Bolton"]
relegated_96 = ["Man City", "QPR", "Bolton"]
promoted_96 = ["Sunderland", "Derby", "Leicester"]
relegated_97 = ["Sunderland", "Middlesbrough", "Notts Forest"]
promoted_97 = ["Bolton", "Barnsley", "Crystal Palace"]
relegated_98 = ["Bolton", "Barnsley", "Crystal Palace"]
promoted_98 = ["Notts Forest", "Middlesbrough", "Charlton"]
relegated_99 = ["Charlton", "Blackburn", "Notts Forest"]
promoted_99 = ["Sunderland", "Bradford", "Watford"]
relegated_00 = ["Wimbledon", "Sheffield Weds", "Watford"]
promoted_00 = ["Charlton", "Man City", "Ipswich"]
relegated_01 = ["Man City", "Coventry", "Bradford"]
promoted_01 = ["Fulham", "Blackburn", "Bolton"]
relegated_02 = ["Ipswich", "Derby", "Leicester"]
promoted_02 = ["Man City", "West Brom", "Birmingham"]
relegated_03 = ["West Ham", "West Brom", "Sunderland"]
promoted_03 = ["Portsmouth", "Leicester", "Wolves"]
relegated_04 = ["Leicester", "Leeds", "Wolves"]
promoted_04 = ["Norwich", "West Brom", "Crystal Palace"]
relegated_05 = ["Crystal Palace", "Norwich", "Southampton"]
promoted_05 = ["Sunderland", "Wigan", "West Ham"]
relegated_06 = ["Birmingham", "West Brom", "Sunderland"]
promoted_06 = ["Reading", "Sheffield United", "Watford"]
relegated_07 = ["Sheffield United", "Charlton", "Watford"]
promoted_07 = ["Sunderland", "Birmingham", "Derby"]
relegated_08 = ["Reading", "Birmingham", "Derby"]
promoted_10 = ["Newcastle", "West Brom", "Blackpool"]
relegated_11 = ["Birmingham", "Blackpool", "West Ham"]
promoted_11 = ["QPR", "Norwich", "Swansea"]
relegated_12 = ["Bolton", "Blackburn", "Wolves"]
promoted_12 = ["Reading", "Southampton", "West Ham"]
relegated_13 = ["Wigan", "Reading", "QPR"]
promoted_09 = ["Wolves", "Birmingham", "Burnley"]
relegated_10 = ["Burnley", "Hull", "Portsmouth"]
relegated_15 = ["Hull", "Burnley", "QPR"]
promoted_15 = ["Bournemouth", "Watford", "Norwich"]
relegated_16 = ["Newcastle", "Norwich", "Aston Villa"]
promoted_13 = ["Cardiff", "Hull", "Crystal Palace"]
relegated_14 = ["Norwich", "Fulham", "Cardiff"]
promoted_14 = ["Leicester", "Burnley", "QPR"]
promoted_08 = ["West Brom", "Stoke", "Hull"]
relegated_09 = ["Newcastle", "Middlesbrough", "West Brom"]
promoted_16 = ["Burnley", "Middlesbrough", "Hull"]
relegated_17 = ["Hull", "Middlesbrough", "Sunderland"]
promoted_17 = ["Newcastle", "Brighton", "Huddersfield"]
relegated_18 = ["Swansea", "Stoke", "West Brom"]
promoted_18 = ["Wolves", "Cardiff", "Fulham"]

relegations = [relegated_95, relegated_96, relegated_97, relegated_98, relegated_99, relegated_00, relegated_01,
               relegated_02, relegated_03, relegated_04, relegated_05, relegated_06, relegated_07, relegated_08,
               relegated_09, relegated_10, relegated_11, relegated_12, relegated_13, relegated_14, relegated_15,
               relegated_16, relegated_17, relegated_18]
promotions = [promoted_95, promoted_96, promoted_97, promoted_98, promoted_99, promoted_00, promoted_01, promoted_02,
              promoted_03, promoted_04, promoted_05, promoted_06, promoted_07, promoted_08, promoted_09, promoted_10,
              promoted_11, promoted_12, promoted_13, promoted_14, promoted_15, promoted_16, promoted_17, promoted_18]


def merge_csv():
    os.chdir(r"C:/Users/cavan/Downloads/PL")  # Change File Path if Necessary
    dflist = []
    for file in range(1, 26):
        filename = str(file)+".csv"
        df1 = pd.read_csv(filename, header=0, engine='python', usecols=(1, 2, 3, 4, 5, 6)).dropna()
        df1['Season'] = file
        dflist.append(df1)
    concatdf = pd.concat(dflist, axis=0, sort=False)
    concatdf.to_csv('C:/Users/cavan/OneDrive/Documents/PL_ML_Predictions/PLR.csv')  # Change File Path if Necessary


def convert_dict():
    for item in seasonDict:
        seasonDict[item]['s'] = datetime.strptime(seasonDict[item]['s'], '%d/%m/%Y')
        seasonDict[item]['e'] = datetime.strptime(seasonDict[item]['e'], '%d/%m/%Y')


def pass_on_form(data):
    for count, releg in enumerate(relegations):
        for index, team in enumerate(releg):
            homeSAF = home_season_a_form(data, team, count+1)
            homeSDF = home_season_d_form(data, team, count+1)
            awaySAF = away_season_a_form(data, team, count+1)
            awaySDF = away_season_d_form(data, team, count+1)
            pass_on_hf(data, promotions[count][index], count+2, homeSAF, homeSDF)
            pass_on_af(data, promotions[count][index], count+2, awaySAF, awaySDF)
    print("Pass On Form:", count / len(relegations) * 100, "%")
    return data


def pass_on_hf(data, team, season, seasonAF, seasonDF):
    data = data.loc[data['HomeTeam'] == team]
    data = slice_dates(data, season)
    data = data.iloc[0:gamesConsidered-1]
    for i, (index, row) in enumerate(data.iterrows()):
        data.at[index, 'homeAF'] = seasonAF
        data.at[index, 'homeDF'] = seasonDF
    return data


def pass_on_af(data, team, season, seasonAF, seasonDF):
    data = data.loc[data['AwayTeam'] == team]
    data = slice_dates(data, season)
    data = data.iloc[0:gamesConsidered-1]
    for i, (index, row) in enumerate(data.iterrows()):
        data.at[index, 'awayAF'] = seasonAF
        data.at[index, 'awayDF'] = seasonDF
    return data


def pass_on_season_form(data):
    for count, releg in enumerate(relegations):
        # Special Case in first season:
        for index, team in enumerate(releg):
            print(team, promotions[count][index])
            homeSAF = home_season_a_form(data, team, count+1)
            print(homeSAF)
            homeSDF = home_season_d_form(data, team, count+1)
            awaySAF = away_season_a_form(data, team, count+1)
            awaySDF = away_season_d_form(data, team, count+1)
            pass_on_s_hf(data, promotions[count][index], count+2, homeSAF, homeSDF)
            pass_on_s_af(data, promotions[count][index], count+2, awaySAF, awaySDF)
    return data


def pass_on_s_hf(data, team, season, seasonAF, seasonDF):
    data = data.loc[data['HomeTeam'] == team]
    data = slice_dates(data, season)
    for i, (index, row) in enumerate(data.iterrows()):
        data.at[index, 'LSHAF'] = seasonAF
        data.at[index, 'LSHDF'] = seasonDF
    return data


def pass_on_s_af(data, team, season, seasonAF, seasonDF):
    data = data.loc[data['AwayTeam'] == team]
    data = slice_dates(data, season)
    for i, (index, row) in enumerate(data.iterrows()):
        data.at[index, 'LSAAF'] = seasonAF
        data.at[index, 'LSADF'] = seasonDF
    return data


def slice_dates(data, season):
    seasonStart = seasonDict[season]['s']
    seasonEnd = seasonDict[season]['e']
    data['Date'] = pd.to_datetime(data.Date, dayfirst=True)
    data = data.set_index(['Date'], drop=True).sort_values(['Date'], ascending=True)
    data = data.loc[seasonStart:seasonEnd]
    return data


def home_season_a_form(data, homeTeam, season):
    if season == 0:
        return 0
    dfcopy = data.copy()
    dfcopy = dfcopy.loc[dfcopy['HomeTeam'] == homeTeam]
    dfcopy = slice_dates(dfcopy, season)
    if len(dfcopy.index) == 0:
        return 0
    homeSAF = dfcopy['homeAF'].sum()/len(dfcopy.index)
    return homeSAF


def home_season_d_form(data, homeTeam, season):
    if season == 0:
        return 0
    dfcopy = data.copy()
    dfcopy = dfcopy.loc[dfcopy['HomeTeam'] == homeTeam]
    dfcopy = slice_dates(dfcopy, season)
    if len(dfcopy.index) == 0:
        return 0
    homeSDF = dfcopy['homeDF'].sum()/len(dfcopy.index)
    return homeSDF


def away_season_a_form(data, awayTeam, season):
    if season == 0:
        return 0
    dfcopy = data.copy()
    dfcopy = dfcopy.loc[dfcopy['AwayTeam'] == awayTeam]
    dfcopy = slice_dates(dfcopy, season)
    if len(dfcopy.index) == 0:
        return 0
    awaySAF = dfcopy['awayAF'].sum()/len(dfcopy.index)
    return awaySAF


def away_season_d_form(data, awayTeam, season):
    if season == 0:
        return 0
    dfcopy = data.copy()
    dfcopy = dfcopy.loc[dfcopy['AwayTeam'] == awayTeam]
    dfcopy = slice_dates(dfcopy, season)
    if len(dfcopy.index) == 0:
        return 0
    awaySDF = dfcopy['awayDF'].sum()/len(dfcopy.index)
    return awaySDF


def home_att_form(data, index):
    dfcopy = data.iloc[index+1:index+gamesConsidered+1]
    dfcheck = data.iloc[index:index+gamesConsidered]
    if (dfcheck['HomeTeam'].nunique() != 1) | (len(dfcopy.index) < 10):
        return 0
    gAve = dfcopy['FTHG'].sum()
    return gAve / gamesConsidered


def home_def_form(data, index):
    dfcopy = data.iloc[index + 1:index + gamesConsidered + 1]
    dfcheck = data.iloc[index:index + gamesConsidered]
    if (dfcheck['HomeTeam'].nunique() != 1) | (len(dfcopy.index) < 10):
        return 0
    gAve = dfcopy['FTAG'].sum()
    return gAve / gamesConsidered


def away_att_form(data, index):
    dfcopy = data.iloc[index + 1:index + gamesConsidered + 1]
    dfcheck = data.iloc[index:index + gamesConsidered]
    if (dfcheck['AwayTeam'].nunique() != 1) | (len(dfcopy.index) < 10):
        return 0
    gAve = dfcopy['FTAG'].sum()
    return gAve / gamesConsidered


def away_def_form(data, index):
    dfcopy = data.iloc[index + 1:index + gamesConsidered + 1]
    dfcheck = data.iloc[index:index + gamesConsidered]
    if (dfcheck['AwayTeam'].nunique() != 1) | (len(dfcopy.index) < 10):
        return 0
    gAve = dfcopy['FTHG'].sum()
    return gAve / gamesConsidered


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


def calc_form(data):
    for i, (index, row) in enumerate(data.iterrows()):
        if data.at[index, 'FTHG'] > data.at[index, 'FTAG']:
            data.at[index, 'result'] = 1  # 1 means a home win
        elif data.at[index, 'FTHG'] < data.at[index, 'FTAG']:
            data.at[index, 'result'] = 2  # 2 means an away win
        else:
            data.at[index, 'result'] = 0  # 0 means a draw
        data.at[index, 'homeAF'] = (home_att_form(data, index))
        data.at[index, 'homeDF'] = (home_def_form(data, index))
        print("Calculate Form:", i/len(data)*50, "%")
    data = data.sort_values(['AwayTeam', 'Date'], ascending=[True, False]).reset_index(drop=True)
    for i, (index, row) in enumerate(data.iterrows()):
        data.at[index, 'awayAF'] = (away_att_form(data, index))
        data.at[index, 'awayDF'] = (away_def_form(data, index))
        print("Calculate Form:", (i/len(data)*50)+50, "%")
    data = data.drop(['FTHG', 'FTAG'], axis=1)
    data = data.sort_values(['Date'], ascending=True).reset_index(drop=True)
    data.to_csv('skips.csv', index =False)
    return data


def last_season_form(data):
    data = data.sort_values(['Date'], ascending=True).reset_index(drop=True)
    for i, (index, row) in enumerate(data.iterrows()):
        hometeam = data.loc[index, 'HomeTeam']
        awayteam = data.loc[index, 'AwayTeam']
        currentseason = data.loc[index, 'Season']
        proregindex = currentseason - 2
        data.at[index, 'LSHAF'] = home_season_a_form(data, hometeam, currentseason - 1)
        data.at[index, 'LSHDF'] = home_season_d_form(data, hometeam, currentseason - 1)
        data.at[index, 'LSAAF'] = away_season_a_form(data, awayteam, currentseason - 1)
        data.at[index, 'LSADF'] = away_season_d_form(data, awayteam, currentseason - 1)
        if currentseason > 1:
            if hometeam in promotions[proregindex]:
                hpassonteam = relegations[proregindex][promotions[proregindex].index(hometeam)]
                data.at[index, 'LSHAF'] = home_season_a_form(data, hpassonteam, currentseason - 1)
                data.at[index, 'LSHDF'] = home_season_d_form(data, hpassonteam, currentseason - 1)
            if awayteam in promotions[proregindex]:
                apassonteam = relegations[proregindex][promotions[proregindex].index(awayteam)]
                data.at[index, 'LSAAF'] = away_season_a_form(data, apassonteam, currentseason - 1)
                data.at[index, 'LSADF'] = away_season_d_form(data, apassonteam, currentseason - 1)
        print("Last Season Form:", int(i/len(data)*100), "%")
        data.to_csv('skip_calc.csv')
    return data


def reorganise_columns(data):
    moveres = data.pop('result')  # Remove result column and store it in df1
    data['result'] = moveres
    data = data.drop(data.index[0:dropteams])#.reset_index(drop=True)
    return data


def refine_columns(data):
    data = data.drop(['FTR'], axis=1)
    data['Date'] = pd.to_datetime(data.Date, dayfirst=True)
    data = data.sort_values(['HomeTeam', 'Date'], ascending=[True, False]).reset_index(drop=True)
    return data


def remove_teams(data):
    data = data.loc[data['HomeTeam'].isin(currentTeams)]
    data = data.loc[data['AwayTeam'].isin(currentTeams)]
    return data


if __name__ == "__main__":
    start = time.time()
    merge_csv()  # Only call if you want to merge all separate .csv files
    os.chdir('C:/Users/cavan/OneDrive/Documents/PL_ML_Predictions')# Change File Path if Necessary
    df = pd.read_csv('PLR.csv', header=0)#.drop(['Unnamed: 0'], axis=1)
    df = refine_columns(df)
    df = df.drop(['Unnamed: 0'], axis=1)
    df = calc_form(df)
    df = pass_on_form(df)
    df = last_season_form(df)
    df = pd.read_csv('skip_calc.csv', header=0)#.drop(['Unnamed: 0'], axis=1)
    df = remove_teams(df)
    dfn = df.copy()
    dfn = dfn.sort_index(ascending=True)
    dfn = dfn.reset_index(drop=True)
    dfn = reorganise_columns(dfn).drop(['Unnamed: 0'], axis=1).reset_index(drop=True)
    dfn.to_csv('neuraldata.csv')
    df = do_dummies(df)
    df = reorganise_columns(df)
    end = time.time()
    print(end-start)
    df = df.sort_values(['Date'], ascending=True)
    df = df.drop(['Date', 'Season'], axis=1).drop(['Unnamed: 0'], axis=1).reset_index(drop=True)
    df.to_csv('wpl.csv', index=False)  # Change File Path if Necessary
