library(zoo)
library(dplyr)
library(tidyr)
library(data.table)

# This file provides all of the cleaning functions used in the decision_tree_R.R file to convert the raw 
# Kaggle data file to a more usable file with custom features and cleaned data

raw_df <- read.csv('NFL Play by PLay 2009-2016 (v3).csv', na.strings = c("",'NA'), stringsAsFactors = FALSE)

df <- dplyr::tbl_df(raw_df)


#----------------------------------


fill_previous_with_na <- function(df) {
  
  # function takes dataframe as a parameter, and fills all NA entries for the selected columns with the value from the previous play
  
  columns_to_fill <- c('down','Possession_Difference','defteam_timeouts_pre','yrdline100')
  df <- fill(df,columns_to_fill)
  
}


# custom features functions section
# -------------------------------------------------------------

defteam_timeouts_pre <- function(df) {
  #calculates how many timeouts the defensive team has left
  df <- df %>% mutate(defteam_timeouts_pre = ifelse(HomeTeam == DefensiveTeam,as.numeric(HomeTimeouts_Remaining_Pre),
                                          ifelse(AwayTeam == DefensiveTeam,as.numeric(AwayTimeouts_Remaining_Pre),'NA')))
  df %>% mutate(defteam_timeouts_pre = as.numeric(defteam_timeouts_pre))
  
  }

InjuryTimeout <- function(df) {
  #determines if timeout is an injury timeout, this is not something we want to predict
  df %>% mutate(Injury_Timeout = ifelse(grepl('njur',desc) & Timeout_Indicator == 1,1,0))
  
}

HomeTeamScore <- function(df) {
  #Calculates Home team's score, not currently in the dataset
  df %>% mutate(HomeTeamScore = ifelse(posteam == HomeTeam,PosTeamScore,DefTeamScore))
  
}

AwayTeamScore <- function(df) {
  #Calculates Home team's score, not currently in the dataset
  df %>% mutate(AwayTeamScore = ifelse(posteam == AwayTeam,PosTeamScore,DefTeamScore))
  
}

Abs_Possession_Difference <- function(df) {
  #Determines how far apart the teams are in terms of possessions, max number of points you can score on a possession is 8
  df %>% mutate(Abs_Possession_Difference = ((AbsScoreDiff - 1) %/% 8) + 1 )
  
  
}

Possession_Difference <- function(df) {
  #Determines how many posessions the team with the ball is winning by
  df %>% mutate(Possession_Difference = Abs_Possession_Difference * (ScoreDiff / ifelse(ScoreDiff == 0,1,abs(ScoreDiff))))
  
  
}

yrdline100_post <- function(df) {
  #determines what yard line the team is following the play, there were some issues with simply shifting the yrdline100 column down
  df %>% mutate(yrdline100_post = yrdline100 - Yards.Gained)
  
}

after_two_minute_warning <- function(df) {
  #simple feature that tells you wheteher it is before or after 2 minute warning
  df %>% mutate(after_two_minute_warning = ifelse(TimeSecs < 121 , 1, 0))
  
}

PotentialClockRunning <- function(df) {
  # This feature determines if the clock is definitely stopped or if on the other hand it is potentially running. Uses some 
  # basic tests that would tell us the clock is definitely stopped.
  
  
  df <- df %>% mutate(PotentialClockRunning = ifelse(PassOutcome == "Incomplete Pass" |
                                                      PlayType %in%  c('Extra Point',
                                                                          'Kickoff',
                                                                          'Spike',
                                                                          'Punt',
                                                                          'No Play',
                                                                          'End of Game',
                                                                          'Quarter End',
                                                                          'Two Minute Warning') |
                                                      TwoPointConv %in% c('Success' , 'Failure') |
                                                      InterceptionThrown == 1 |
                                                      Fumble == 1 |
                                                      sp == 1
                                                     ,0,1)) 

  df <- df %>% mutate(PotentialClockRunning = ifelse(is.na(PotentialClockRunning),1,PotentialClockRunning))  
  
  }


first_down_post <- function(df) {
  # Determiens whether or not the play resulted in a first down. The First.Down column was incomplete and needed to be supplemented
  # with my custom first down conditions
  
  df %>% mutate(first_down_post = ifelse(FirstDown == 1,1,
                                               ifelse(all(Yards.Gained > ydstogo,
                                                          ydstogo > 0,
                                                          Accepted.Penalty == 0,
                                                          sp == 0,
                                                          Challenge.Replay == 0,
                                                          PlayType != "No Play"),1,0)))
  
}


add_custom_features <- function(df) {
  df <- defteam_timeouts_pre(df)
  df <- InjuryTimeout(df)
  df <- HomeTeamScore(df)
  df <- AwayTeamScore(df)
  df <- Abs_Possession_Difference(df)
  df <- Possession_Difference(df)
  df <- yrdline100_post(df)
  df <- after_two_minute_warning(df)
  df <- PotentialClockRunning(df)
  df <- first_down_post(df)
    
}

#---------------------------------------------


add_shifted_columns <- function (df) {
  
  # this function adds certain values from the following entry in the data to help determine the outcome of the current play
  # an example of this is to actually see when teams call timeouts, as the raw data records timeouts as their own row

  columns_to_be_shifted = c('Timeout_Indicator','Timeout_Team','posteam_timeouts_pre','defteam_timeouts_pre',
                            'HomeTimeouts_Remaining_Pre','AwayTimeouts_Remaining_Pre',
                            'HomeTimeouts_Remaining_Post','AwayTimeouts_Remaining_Post',
                            'Injury_Timeout',
                            'Challenge.Replay',
                            'down','ydstogo')
  
  shift_columns <- function (df,column_list) {
    
    for (column in column_list){
      
      new_name <- paste0(column,'_s')
      
      df[[new_name]] <- with(df, shift(df[[column]],type = "lead",1))
      
      }  
      
      df
    
  }  
  
  shift_columns(df,columns_to_be_shifted)
  
}


#---------------------------------------------
#Resulting Down function to calculate what the down of the following play is

add_resulting_down <- function(df) {
  
  df['down'][is.na(df['down'])] <- 0
  df['down_s'][is.na(df['down_s'])] <- 0 
    
  
  
  mutate(df,down_post_play =  
                ifelse(first_down_post == 1,1,
                  
                ifelse(all(down == 4,down_s == 4, grepl('enalty',desc)),4,

                ifelse(down == 4,1,
                
                ifelse(down == 'TRUE' , down,

                down + 1)
                 ))))
  
}


#--------------------------------------------

#Timeout Label functions

add_timeout_label <- function(df) {
  
  df %>% mutate(Timeout_Label = ifelse(Challenge.Replay_s==1,0,
                                       ifelse(Injury_Timeout_s==1,0,
                                              ifelse(PotentialClockRunning==0,0,
                                                     ifelse(Timeout_Indicator==1,0,
                                                            Timeout_Indicator_s)))))
  
  
}


def_timeout_label <- function(df) {
  
  df %>% mutate(Def_Timeout_Label = ifelse(Timeout_Label==1 & DefensiveTeam==Timeout_Team_s,1,0))
  
}



