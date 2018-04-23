library(gdata)
library(caret)
library(rpart.plot)
library(e1071)
source(nfl_data_cleaning_R)

#read raw data
raw_df <- read.csv('NFL Play by PLay 2009-2016 (v3).csv', na.strings = c("",'NA'), stringsAsFactors = FALSE)

df <- dplyr::tbl_df(raw_df)

#cleaning functions from nfl_data_cleaning_R file

df <- add_custom_features(df)
df <- fill_previous_with_na(df)
df <- add_shifted_columns(df)
df <- add_resulting_down(df)
df <- add_timeout_label(df)
df <- def_timeout_label(df)
nfl_data <- df

#comment out next line if you don't already have clean data file

nfl_data=read.csv('clean_nfl_data.csv',na.strings=c("", "NA"))

#convert label to factor in order to do classification

nfl_data$Def_Timeout_Label = as.factor(nfl_data$Def_Timeout_Label)

#Filter data to end of game plays

filtered_data <- nfl_data[nfl_data$TimeSecs < 300 & nfl_data$TimeSecs > 0,]

#Narrow down the data frame to only columns that could be potentiall relevant

relevant_columns = c('Date','GameID','HomeTeam','AwayTeam','posteam','DefensiveTeam',
                     'PosTeamScore','DefTeamScore','ScoreDiff','AbsScoreDiff','Possession_Difference',
                     'qtr','time','TimeSecs','after_two_minute_warning','PlayTimeDiff','down','ydstogo','yrdline100',
                     'down_s','down_post_play','ydstogo_s','yrdline100_post',
                     'desc','PlayType','PassOutcome','RushAttempt',
                     'Accepted.Penalty','PenalizedTeam',
                     'Timeout_Label','Pos_Timeout_Label','Def_Timeout_Label',
                     'Timeout_Indicator_s','Timeout_Team_s','posteam_timeouts_pre_s','defteam_timeouts_pre_s',
                     'HomeTimeouts_Remaining_Pre_s','AwayTimeouts_Remaining_Pre_s',
                     'HomeTimeouts_Remaining_Post_s','AwayTimeouts_Remaining_Post_s',
                     'PotentialClockRunning')
                    

filtered_data <- filtered_data[,relevant_columns]


# ------ Defensive Timeout Model First ------
# Since the strategy for calling offensive and defensive timeouts is very different, we will need to train different
# classifiers for offensive and defensive timeouts. As of now, this model only predicts defensive timeouts
# The following lines filter rows based on subjective situations where we would most likely not want to predict a
# defensive timeout

filtered_data <- filtered_data[filtered_data$PotentialClockRunning == 1,]
filtered_data <- filtered_data[is.na(filtered_data$PenalizedTeam),]
filtered_data <- filtered_data[filtered_data$ScoreDiff < 17 & filtered_data$ScoreDiff > -9 ,]

#remove NA rows
filtered_data <-  filtered_data[rowSums(is.na(filtered_data)) != ncol(filtered_data),]

# Select features and label for classifier training, starting with less features and seeing the impact of adding
# other features

training_columns <- c('TimeSecs',
                      'after_two_minute_warning',
                      'Possession_Difference',
                      'ScoreDiff',
                      'down_post_play',
                      'yrdline100_post',
                      'defteam_timeouts_pre_s',
                      'Def_Timeout_Label')

Def_df <- filtered_data[,training_columns]

#drop NA rows
Def_df <- na.omit(Def_df)




#slice into testing and training sets
set.seed(3003)
intrain <- caret::createDataPartition(y=Def_df$Def_Timeout_Label,p = 0.7, list = FALSE)
training <- Def_df[intrain,]
testing <- Def_df[-intrain,]

#train decision tree classifier
trctrl <- caret::trainControl(method = 'repeatedcv', number = 10, repeats = 3)
set.seed(3333)
decision_tree_fit <- train(Def_Timeout_Label ~., data = training, method = 'rpart', 
                                  parms = list(split = 'gini'),
                                  trControl = trctrl,
                                  tuneLength = 10)

#make predictions and test accuracy
test_pred <- predict(decision_tree_fit,newdata = testing)
confusionMatrix(test_pred,testing$Def_Timeout_Label)


