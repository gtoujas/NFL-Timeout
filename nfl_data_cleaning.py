import pandas as pd
import csv

##bring in play by play data to pandas dataframe

#shift events that happen in their own row to the previous row,
# eg. when a timeout is called, we want that information with the row that corresponds to the play directly before the timeout, rather than on its own row
# will denote with a '_s' for shifted up
# look at negative timeouts and injury and all

def add_shifted_columns(dataframe):

    print "Adding post-play data to each play"
    columns_to_be_shifted = ['Timeout_Indicator','Timeout_Team','posteam_timeouts_pre',
                             'HomeTimeouts_Remaining_Pre','AwayTimeouts_Remaining_Pre',
                             'HomeTimeouts_Remaining_Post','AwayTimeouts_Remaining_Post',
                             'Injury_Timeout',
                             'Challenge.Replay']

    for column_string in columns_to_be_shifted:
        dataframe[str(column_string+'_s')] = dataframe[column_string].shift(-1)

    return dataframe

#add custom columns for new features

# add injury timeout feature

def InjuryTimeout(row):
    if str(row['desc']).__contains__('njury') and row['Timeout_Indicator'] == 1:
        InjuryTimeout = 1
    else:
        InjuryTimeout = 0
    return InjuryTimeout

#get HomeTeamScore

def HomeTeamScore(row):
    if row['posteam'] == row['HomeTeam']:
        HomeTeamScore = row['PosTeamScore']
    else:
        HomeTeamScore = row['DefTeamScore']
    return HomeTeamScore

#get AwayTeamScore

def AwayTeamScore(row):
    if row['posteam'] == row['AwayTeam']:
        AwayTeamScore = row['PosTeamScore']
    else:
        AwayTeamScore = row['DefTeamScore']
    return AwayTeamScore

#get how many possession game - can score max of 8 points per possession

def Possession_Difference(row):
    score_diff = row['AbsScoreDiff']
    poss = ((score_diff-1)//8) + 1
    return poss

#get PotentialClockRunning
#logic for when the clock is definitely stopped ---- incomplete pass, spike, after scoring play, turnover, kickoff/punt,
#                                                    some accepted penalties,- look up rules on this (will add in future version),
#                                                    think about potential issues around the 2 minute warning?


def PotentialClockRunning(row):
    list_truth_conditions = [row['PassOutcome']=='Incomplete Pass',
                             row['PlayType'] in ['Extra Point','Kickoff','Spike','Punt'],
                             row['TwoPointConv'] in ['Success','Failure'],
                             ]

    if any(list_truth_conditions)==True:
        PotentialClockRunning = 0
    else:
        PotentialClockRunning = 1
    return PotentialClockRunning


#apply the functions to add custom features

def add_custom_features(dataframe):
    print "Adding custom features"
    print "Adding Injury Timeout"
    dataframe['Injury_Timeout'] = dataframe.apply(lambda row: InjuryTimeout(row), axis=1)
    print "Adding Home Team Score"
    dataframe['HomeTeamScore'] = dataframe.apply(lambda row: HomeTeamScore(row), axis=1)
    print "Adding Away Team Score"
    dataframe['AwayTeamScore'] = dataframe.apply(lambda row: AwayTeamScore(row), axis=1)
    print "Adding Possession Score Difference"
    dataframe['Possession_Difference'] = dataframe.apply(lambda row: Possession_Difference(row), axis=1)
    print "Adding Potential Clock Running Feature"
    dataframe['PotentialClockRunning'] = dataframe.apply(lambda row: PotentialClockRunning(row), axis=1)

    return dataframe

#method to remove games that have plays with negative timeouts left because of whacky NFL injury and timeout rule
def remove_games_with_negative_timeouts(dataframe):
    print "Removing games with negative timeout values"
    plays_with_negative_timeouts_left = dataframe.query('(HomeTimeouts_Remaining_Post < 0) | (AwayTimeouts_Remaining_Post < 0)')
    games_to_remove = plays_with_negative_timeouts_left.GameID.unique()
    #dataframe = dataframe.drop(dataframe[dataframe.GameID in games_to_remove].index)
    dataframe = dataframe[~dataframe.GameID.isin(games_to_remove)]

    return dataframe


#method that creates the timeout label, eliminates some timeouts that we know we want excluded from training the classifier
#types of timeouts eliminated include challenge timeouts and injury timeouts, and timeouts taken directly after the other team took one

def Timeout_Label(row):

    Timeout_Label = row['Timeout_Indicator_s']

    if row['Challenge.Replay_s'] == 1:
        Timeout_Label = 0

    if row['Injury_Timeout_s'] == 1:
        Timeout_Label = 0

    if row['PotentialClockRunning'] == 0:
        Timeout_Label = 0

    if row['Timeout_Indicator'] == 1:
        Timeout_Label = 0

    return Timeout_Label


#method that adds the timeout label

def add_timeout_label(dataframe):
    print "Adding timeout label"
    dataframe['Timeout_Label'] = dataframe.apply(lambda row: Timeout_Label(row), axis=1)

    return dataframe


def main():

    return

if __name__ == "__main__":
    main()