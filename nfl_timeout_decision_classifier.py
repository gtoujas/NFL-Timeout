from nfl_data_cleaning import *
import pandas as pd


"""

This file first uses the methods defined in nfl_data_cleaning.py to clean the raw Kaggle data, and then trains certain
Decision Tree Classifiers in an attempt to replicate the decision making process that an NFL coach undergoes when
determining how best to use timeouts to stop the clock.

"""




"""
Import raw Kaggle data into dataframe
"""

initial_df = pd.read_csv('NFL Play by Play 2009-2016 (v3).csv')



def cleandata(dataframe):

    """
    Function that combines all cleaning functions imported from nfl_data_cleaning.py to clean the data in one step
    """

    cleaned_df = add_custom_features(dataframe)
    cleaned_df = fill_na_with_previous_value(cleaned_df)
    cleaned_df = add_shifted_columns(cleaned_df)
    cleaned_df = add_resulting_down(cleaned_df)
    cleaned_df = add_timeout_label(cleaned_df)
    cleaned_df = remove_games_with_negative_timeouts(cleaned_df)
    cleaned_df.to_csv('clean_nfl_data.csv',sep=',',index=False)

    return cleaned_df

"""
Comment out the next line if you already have a 'clean_nfl_data.csv' file
"""
cleaned_df = cleandata(initial_df)


"""
Save clean data file to avoid cleaning every time you make changes to the classifiers
"""
cleaned_df = pd.read_csv('clean_nfl_data.csv',sep=',')



"""
Limit plays to only the end of the game, where teams are most likely to be using timeouts to stop the clock
Ignore 1st half and overtime for now
"""

time_shortened_df = cleaned_df.query('(300 > TimeSecs > 0)')


"""
Narrow down the dataframe to only the columns that could even potentially be relevant for timeout decision making
"""

relevant_columns = ['Date','GameID','HomeTeam','AwayTeam','posteam','DefensiveTeam',
                    'PosTeamScore','DefTeamScore','ScoreDiff','AbsScoreDiff','Possession_Difference',
                    'qtr','time','TimeSecs','after_two_minute_warning','PlayTimeDiff','down','ydstogo','yrdline100',
                    'down_s','down_post_play','ydstogo_s','yrdline100_post',
                    'desc','PlayType','PassOutcome','RushAttempt',
                    'Accepted.Penalty','PenalizedTeam',
                    'Timeout_Label','Pos_Timeout_Label','Def_Timeout_Label',
                    'Timeout_Indicator_s','Timeout_Team_s','posteam_timeouts_pre_s','defteam_timeouts_pre_s',
                    'HomeTimeouts_Remaining_Pre_s','AwayTimeouts_Remaining_Pre_s','HomeTimeouts_Remaining_Post_s','AwayTimeouts_Remaining_Post_s',
                    'PotentialClockRunning'
                    ]

first_relevant_df = pd.DataFrame(time_shortened_df,columns=relevant_columns)

"""

------ Defensive Timeout Model First ------

Since the strategy for calling offensive and defensice timeouts is very different, we will need to train different
classifiers for offensive and defensive timeouts. As of now, this model only predicts defensive timeouts


The following lines filter rows based on subjective situations where we would most likely not want to predict a defensive timeout
"""


filtered_df = first_relevant_df.query('PotentialClockRunning==1')
filtered_df = filtered_df[filtered_df['PenalizedTeam'].isnull()]
filtered_df = filtered_df.query('(-9<ScoreDiff<17)')
filtered_df.to_csv('filtered_def.csv',sep=',',index=False)



"""
NEED TO FIX ydstogo_s, then put it back in the model, issue is that all timeouts occur when ydstogo_s is 0 due to how data is set up, remove ydstogo_s from shifted columns and from learning model
"""


Def_df = pd.DataFrame(filtered_df,columns=['TimeSecs','after_two_minute_warning','Possession_Difference','ScoreDiff','down_post_play','yrdline100_post','defteam_timeouts_pre_s','Def_Timeout_Label'])


#need to drop rows with missing values if there are any

nulls_df = Def_df[pd.isnull(Def_df).any(axis=1)]
na_rows = nulls_df.__len__()
print('Rows with null values to be deleted: ' + str(na_rows))

Def_df=Def_df.dropna(axis=0)


#split into X vs y

Def_X_df = Def_df.drop(['Def_Timeout_Label'], axis=1)
Def_y_df = Def_df['Def_Timeout_Label']

#split into training vs test sets - consider splitting on random games instead of every single play, not sure how to think about this yet

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(Def_X_df,Def_y_df, test_size=0.25,random_state = 1)

#train simple decision treee classifier on training set

"""

Notes for Decision Tree Classifier

max_depth set low to better visualize whats going
min_sample_leaf set at 5 per sklearn
sample_weight is set to 1:1, may want to try to vary this, read more about sample weights
seems like AdaBoost works on normalizing sample weights?

adding class_weight makes the tree make a lot more sense now, more accurate as well

try min_impurity_decrease with a deep tree?

its possible that timesecs is causing some overfitting, look into having a very simple before and after two minute warning? or something more along the lines of time buckets?
"""

from sklearn import tree
clf = tree.DecisionTreeClassifier(max_depth = 8,
                                  min_samples_leaf=5,
                                  class_weight = 'balanced',
                                  min_impurity_split = .1)
clf = clf.fit(X_train,y_train)

# make predictions on test set and compute f1 accuracy due to imbalanced classes - high percentage of observations will be no timeout
pred = clf.predict(X_test)
from sklearn.metrics import f1_score
f1_acc = f1_score(y_test,pred)

print "F1 Accuracy is --- " + str(f1_acc)

"""
Use Grid Search to find the best parameters of the decision tree classifier to maximize the f1_score
"""
from sklearn.model_selection import GridSearchCV
clf_grid = tree.DecisionTreeClassifier()
parameters = {'min_samples_split':[2,5,10,15,20,50,100,200,500],'min_samples_leaf':[1,2,4,8,16,32,64]}
clf_grid = GridSearchCV(clf_grid,parameters,scoring='f1')
clf_grid.fit(X_train,y_train)

clf_grid = clf_grid.best_estimator_

from sklearn.metrics import classification_report
grid_pred = clf_grid.predict(X_test)
print classification_report(y_test,grid_pred,target_names=['No Timeout','Timeout'])


"""

Use AdaBoost to boost the best estimator found in the grid search

"""

from sklearn.ensemble import AdaBoostClassifier

ada_clf = AdaBoostClassifier(clf_grid)
ada_clf.fit(X_train,y_train)
ada_pred = ada_clf.predict(X_test)

print "AdaBoost Classification Report -----"
print classification_report(y_test,ada_pred,target_names=['No Timeout','Timeout'])


"""

Attempt at Random Forest Classifier with grid search

"""

from sklearn.ensemble import RandomForestClassifier

rfc_grid = RandomForestClassifier()
rfc_param = parameters
rfc_grid = GridSearchCV(rfc_grid,parameters,scoring='f1')
rfc_grid.fit(X_train,y_train)
rfc_grid = rfc_grid.best_estimator_

rfc_pred = rfc_grid.predict(X_test)

print "Random Forest Classification Report -----"
print classification_report(y_test,rfc_pred,target_names=['No Timeout','Timeout'])




#export tree to graphical format so that we can visualize it


import graphviz




graph_dot = tree.export_graphviz(clf,
                                 out_file = "graph.txt",
                                 feature_names=list(Def_X_df),
                                 class_names=['No Timeout','Timeout'],
                                 filled=True,
                                 rounded = True)
graph = graphviz.Source(graph_dot)

#graph.render("graph")

