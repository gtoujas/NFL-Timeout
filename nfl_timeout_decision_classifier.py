from nfl_data_cleaning import *
import pandas as pd


#import raw kaggle data into dataframe
initial_df = pd.read_csv('NFL Play by Play 2009-2016 (v3).csv')

#use functions in cleaning file to clean the raw data from kaggle
cleaned_df = add_custom_features(initial_df)
cleaned_df = add_shifted_columns(cleaned_df)
cleaned_df = add_timeout_label(cleaned_df)
cleaned_df = remove_games_with_negative_timeouts(cleaned_df)


cleaned_df.to_csv('clean_nfl_data.csv',sep=',',index=False)


#limit plays to only the second half of both the second quarter and fourth quarter, where teams are most likely to be using timeouts to stop the clock
#ignore overtime for now


time_shortened_df = cleaned_df.query('(450 > TimeSecs > 0) | (2250 > TimeSecs > 1800)')


#keep only the columns we think could possibly be relevant for now

relevant_columns = ['Date','GameID','HomeTeam','AwayTeam','posteam','DefensiveTeam',
                    'PosTeamScore','DefTeamScore','ScoreDiff','AbsScoreDiff',
                    'qtr','time','TimeSecs','PlayTimeDiff','down','ydstogo','yrdline100',
                    'desc','PlayType','PassOutcome','RushAttempt',
                    'Accepted.Penalty','PenalizedTeam',
                    'Timeout Label','Timeout_Indicator_s','Timeout_Team_s','posteam_timeouts_pre_s',
                    'HomeTimeouts_Remaining_Pre_s','AwayTimeouts_Remaining_Pre_s','HomeTimeouts_Remaining_Post_s','AwayTimeouts_Remaining_Post_s'
                    ]

first_relevant_df = pd.DataFrame(time_shortened_df,columns=relevant_columns)

