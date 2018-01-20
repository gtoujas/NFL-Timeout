import pandas as pd

"""

This file provides all of the cleaning functions used in the nfl_timeout_decision_classifier.py file to convert the raw
Kaggle data file to a more usable file with custom features and cleaned data.


"""


def fill_na_with_previous_value(dataframe):

    """
    function takes raw data dataframe as a parameter, and fills all NA entries with the value from a previous play
    """

    print "Filling empty values"

    columns_to_fill = ['down','Possession_Difference','defteam_timeouts_pre','yrdline100'
                        ]
    for column in columns_to_fill:
        dataframe[str(column)] = dataframe[str(column)].fillna(method='ffill')

    return dataframe

def add_shifted_columns(dataframe):

    """
    this function adds certain values from the following entry in the data to help determine what the outcome of the
    current play was

    an example of this is actually to see when teams call timeouts, since timeouts are recorded as their own "play" in
    the data
    """


    print "Adding post-play data to each play"
    columns_to_be_shifted = ['Timeout_Indicator','Timeout_Team','posteam_timeouts_pre','defteam_timeouts_pre',
                             'HomeTimeouts_Remaining_Pre','AwayTimeouts_Remaining_Pre',
                             'HomeTimeouts_Remaining_Post','AwayTimeouts_Remaining_Post',
                             'Injury_Timeout',
                             'Challenge.Replay',
                             'down','ydstogo']

    for column_string in columns_to_be_shifted:
        dataframe[str(column_string+'_s')] = dataframe[column_string].shift(-1)

    return dataframe





def InjuryTimeout(row):

    """
    functions determines if a timeout is an injury timeout, this is not something we want to attempt to predict
    """

    if str(row['desc']).__contains__('njury') and row['Timeout_Indicator'] == 1:
        InjuryTimeout = 1
    else:
        InjuryTimeout = 0
    return InjuryTimeout



def HomeTeamScore(row):

    """
    Calculates the home team's score, not currently in the dataset
    """

    if row['posteam'] == row['HomeTeam']:
        HomeTeamScore = row['PosTeamScore']
    else:
        HomeTeamScore = row['DefTeamScore']
    return HomeTeamScore



def AwayTeamScore(row):

    """
    Calculates away team's score, not currently in the dataset
    """

    if row['posteam'] == row['AwayTeam']:
        AwayTeamScore = row['PosTeamScore']
    else:
        AwayTeamScore = row['DefTeamScore']
    return AwayTeamScore



def Abs_Possession_Difference(row):

    """
    Determines how far the team's score's are apart in terms of possessions, max number of points you can score on a
    possession is 8. Calculates in absolute terms, similar to the absolute difference in team's score's
    """

    abs_score_diff = row['AbsScoreDiff']
    abs_poss_diff = ((abs_score_diff-1)//8) + 1
    return abs_poss_diff

def Possession_Difference(row):

    """
    Calculates how many possessions the team with the ball is winning by, max number of points you can score on a
    possession is 8.
    """

    score_diff = row['ScoreDiff']
    poss_diff = (((abs(score_diff)-1)//8) + 1) * (score_diff/(max(1,abs(score_diff))))
    return poss_diff


def defteam_timeouts_pre(row):

    """
    Calculates how many timeouts the defensive team has left
    """

    defteam_timeouts_pre = None
    if row['HomeTeam']==row['DefensiveTeam']:
        defteam_timeouts_pre=row['HomeTimeouts_Remaining_Pre']
    if row['AwayTeam']==row['DefensiveTeam']:
        defteam_timeouts_pre=row['AwayTimeouts_Remaining_Pre']
    return defteam_timeouts_pre



def yrdline100_post(row):

    """
    Determines what yard line the play ends on, shifting the yrdline100 column had issues with entries that had issues
    """

    yrdline100_post = row['yrdline100'] - row['Yards.Gained']
    return yrdline100_post


def after_two_minute_warning(row):

    """
    Simple feature that determines whether it is before or after the two minute warning, did not help with accuracy
    """

    if row['TimeSecs'] < 121:
        after_two_minute_warning = 1
    else:
        after_two_minute_warning = 0

    return after_two_minute_warning


def first_down_post(row):

    """
    Determines whether or not the play resulted in a first down. The First.Down column was incomplete and did not signal
    all first downs. Used in the down_post_play function below.
    """

    first_down_conditions = [row['Yards.Gained'] > row['ydstogo'],
                             row['ydstogo'] > 0,
                             row['Accepted.Penalty'] == 0,
                             row['sp'] == 0,
                             row['Challenge.Replay'] == 0,
                             row['PlayType'] != "No Play"
                             ]

    if row['FirstDown'] == 1:
        first_down_post = 1
    elif all(first_down_conditions) == True:
        first_down_post = 1
    else:
        first_down_post = 0

    return first_down_post



def down_post_play(row):

    """
    Determines what down it is after the play.
    """

    consecutive_fourth_down_conditions = [row['down'] == 4,
                                          row['down_s'] == 4,
                                          str(row['desc']).__contains__('enalty')]

    if first_down_post == 1:
        down_post_play = 1

    elif all(consecutive_fourth_down_conditions) == True:
        down_post_play = 4

    elif row['down'] == 4:
        down_post_play = 1

    elif pd.isnull(row['down']):
        down_post_play = ""

    else:
        down_post_play = int(row['down']) + 1

    return down_post_play



def PotentialClockRunning(row):

    """
    Determines whether the clock is definitely stopped or if it may be running. Any timeout called after a play where
    the clock stopped is not one that we want to train on or try to predict.
    """


    list_truth_conditions = [row['PassOutcome']=='Incomplete Pass',
                             row['PlayType'] in ['Extra Point','Kickoff','Spike','Punt','No Play','End of Game','Quarter End','Two Minute Warning'],
                             row['TwoPointConv'] in ['Success','Failure'],
                             1 in [row['InterceptionThrown'],row['Fumble']],
                             row['sp']==1
                             ]

    if any(list_truth_conditions)==True:
        PotentialClockRunning = 0
    else:
        PotentialClockRunning = 1
    return PotentialClockRunning



#apply the functions to add custom features

def add_custom_features(dataframe):

    """
    Combines all the feature functions into one function to easily implement in the classifier script.
    """

    print "Adding custom features"
    print "Adding defensive team timeouts left pre"
    dataframe['defteam_timeouts_pre'] = dataframe.apply(lambda row: defteam_timeouts_pre(row), axis=1)
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
    print "Adding post play yrdline100"
    dataframe['yrdline100_post'] = dataframe.apply(lambda row: yrdline100_post(row), axis=1)
    print "Adding post 2 Minute Warning Feature"
    dataframe['after_two_minute_warning'] = dataframe.apply(lambda row: after_two_minute_warning(row), axis=1)
    print "Adding Potential Clock Running Feature"
    dataframe['PotentialClockRunning'] = dataframe.apply(lambda row: PotentialClockRunning(row), axis=1)

    return dataframe



def add_resulting_down(dataframe):

    """
    Adds the down that the play results in on it's own since it relies on down_s column.
    """

    print "Adding Down Post Play"
    dataframe['down_post_play'] = dataframe.apply(lambda row: down_post_play(row), axis=1)

    return dataframe


def remove_games_with_negative_timeouts(dataframe):

    """
    Very small amount of games have negative timeouts due to whacky NFL rule about injury timeouts
    """

    print "Removing games with negative timeout values"
    plays_with_negative_timeouts_left = dataframe.query('(HomeTimeouts_Remaining_Post < 0) | (AwayTimeouts_Remaining_Post < 0)')
    games_to_remove = plays_with_negative_timeouts_left.GameID.unique()
    dataframe = dataframe[~dataframe.GameID.isin(games_to_remove)]

    return dataframe



def Timeout_Label(row):

    """
     Function that rules out timeouts that we know we don't want to predict. These include timeouts lost on challenges,
     injury timeouts, timeouts called when the clock is running, etc
    """

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

    """
    From the timeout Label funciton, this determines if the timeout we want to predict was called by the offensive team
    """

    Pos_Timeout_Label = 0
    if row['Timeout_Label']==1 and row['posteam']==row['Timeout_Team_s']:
            Pos_Timeout_Label = 1

    return Pos_Timeout_Label

def Def_Timeout_Label(row):

    """
     From the timeout Label funciton, this determines if the timeout we want to predict was called by the defensive team
    """

    Def_Timeout_Label = 0
    if row['Timeout_Label']==1 and row['DefensiveTeam']==row['Timeout_Team_s']:
            Def_Timeout_Label = 1

    return Def_Timeout_Label

#method that adds the timeout labels

def add_timeout_label(dataframe):

    """
    One function for the classifier file to use to add all three timeout labels, the global timeout label and
    the offenzive and defensive ones
    """

    print "Adding timeout labels"
    dataframe['Timeout_Label'] = dataframe.apply(lambda row: Timeout_Label(row), axis=1)
    dataframe['Pos_Timeout_Label'] = dataframe.apply(lambda row: Pos_Timeout_Label(row), axis=1)
    dataframe['Def_Timeout_Label'] = dataframe.apply(lambda row: Def_Timeout_Label(row), axis=1)

    return dataframe


def main():

    return

if __name__ == "__main__":
    main()