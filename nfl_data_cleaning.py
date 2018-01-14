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
                             'Challenge.Replay',
                             'down','ydstogo']

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

def Abs_Possession_Difference(row):
    abs_score_diff = row['AbsScoreDiff']
    abs_poss_diff = ((abs_score_diff-1)//8) + 1
    return abs_poss_diff

def Possession_Difference(row):
    score_diff = row['ScoreDiff']
    poss_diff = (((abs(score_diff)-1)//8) + 1) * (score_diff/(max(1,abs(score_diff))))
    return poss_diff


#get PotentialClockRunning
#logic for when the clock is definitely stopped ---- incomplete pass, spike, after scoring play, turnover, kickoff/punt,
#                                                    some accepted penalties,- look up rules on this (will add in future version),
#                                                    think about potential issues around the 2 minute warning?


def PotentialClockRunning(row):
    list_truth_conditions = [row['PassOutcome']=='Incomplete Pass',
                             row['PlayType'] in ['Extra Point','Kickoff','Spike','Punt'],
                             row['TwoPointConv'] in ['Success','Failure'],
                             1 in [row['InterceptionThrown'],row['Fumble']]
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
    print "Adding Abs Possession Score Difference"
    dataframe['Abs_Possession_Difference'] = dataframe.apply(lambda row: Abs_Possession_Difference(row), axis=1)
    print "Adding Possession Score Difference"
    dataframe['Possession_Difference'] = dataframe.apply(lambda row: Abs_Possession_Difference(row), axis=1)
    print "Adding Potential Clock Running Feature"
    dataframe['PotentialClockRunning'] = dataframe.apply(lambda row: PotentialClockRunning(row), axis=1)

    return dataframe

#method to remove games that have plays with negative timeouts left because of whacky NFL injury and timeout rule
def remove_games_with_negative_timeouts(dataframe):
    print "Removing games with negative timeout values"
    plays_with_negative_timeouts_left = dataframe.query('(HomeTimeouts_Remaining_Post < 0) | (AwayTimeouts_Remaining_Post < 0)')
    games_to_remove = plays_with_negative_timeouts_left.GameID.unique()
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

#Need two different labels for whether it is an offensive timeout or a defensive timeout

def Pos_Timeout_Label(row):
    Pos_Timeout_Label = 0
    if row['Timeout_Label']==1 and row['posteam']==row['Timeout_Team_s']:
            Pos_Timeout_Label = 1

    return Pos_Timeout_Label

def Def_Timeout_Label(row):
    Def_Timeout_Label = 0
    if row['Timeout_Label']==1 and row['DefensiveTeam']==row['Timeout_Team_s']:
            Def_Timeout_Label = 1

    return Def_Timeout_Label

#method that adds the timeout labels

def add_timeout_label(dataframe):
    print "Adding timeout labels"
    dataframe['Timeout_Label'] = dataframe.apply(lambda row: Timeout_Label(row), axis=1)
    dataframe['Pos_Timeout_Label'] = dataframe.apply(lambda row: Pos_Timeout_Label(row), axis=1)
    dataframe['Def_Timeout_Label'] = dataframe.apply(lambda row: Def_Timeout_Label(row), axis=1)

    return dataframe


def main():

    return

if __name__ == "__main__":
    main()