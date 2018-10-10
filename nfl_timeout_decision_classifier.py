from nfl_data_cleaning import *
import pandas as pd
import numpy as np

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
                    'HomeTimeouts_Remaining_Pre_s','AwayTimeouts_Remaining_Pre_s',
                    'HomeTimeouts_Remaining_Post_s','AwayTimeouts_Remaining_Post_s',
                    'PotentialClockRunning'
                    ]

first_relevant_df = pd.DataFrame(time_shortened_df,columns=relevant_columns)

"""

------ Defensive Timeout Model First ------

Since the strategy for calling offensive and defensive timeouts is very different, we will need to train different
classifiers for offensive and defensive timeouts. As of now, this model only predicts defensive timeouts


The following lines filter rows based on subjective situations where we would most likely not want to predict a
defensive timeout
"""


filtered_df = first_relevant_df.query('PotentialClockRunning==1')
filtered_df = filtered_df[filtered_df['PenalizedTeam'].isnull()]
filtered_df = filtered_df.query('(-9<ScoreDiff<17)')
filtered_df.to_csv('filtered_def.csv',sep=',',index=False)



"""
Select features and label for classifier training, starting with less features and seeing the impact of adding
other features
"""


Def_df = pd.DataFrame(filtered_df,columns=['TimeSecs',
                                           'after_two_minute_warning',
                                           'Possession_Difference',
                                           'ScoreDiff',
                                           'down_post_play',
                                           'yrdline100_post',
                                           'defteam_timeouts_pre_s',
                                           'Def_Timeout_Label'])


"""
Identify how many rows with NaN's there are and drop them from the dataframe
"""

nulls_df = Def_df[pd.isnull(Def_df).any(axis=1)]
na_rows = nulls_df.__len__()
print('Rows with null values to be deleted: ' + str(na_rows))

Def_df=Def_df.dropna(axis=0)

"""
Split dataframe into X and y, and apply train_test_split to split out the testing set
"""


Def_X_df = Def_df.drop(['Def_Timeout_Label'], axis=1)
Def_y_df = Def_df['Def_Timeout_Label']



from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(Def_X_df,Def_y_df, test_size=0.25,random_state = 1)



"""

Notes for Decision Tree Classifier

max_depth set low to better visualize whats going
min_sample_leaf set at 5 per sklearn
sample_weight is set to 1:1, may want to try to vary this, read more about sample weights
seems like AdaBoost works on normalizing sample weights?

adding class_weight makes the tree make a lot more sense now, more accurate as well

try min_impurity_decrease with a deep tree?

its possible that timesecs is causing some overfitting, look into having a very simple before and after
two minute warning? or something more along the lines of time buckets?

bring in more features, but let it split only on the most important x features

"""

from sklearn import tree
clf = tree.DecisionTreeClassifier(criterion = 'entropy',
                                  splitter = 'best',
                                  min_impurity_split = .05,
                                  min_samples_leaf=8,
                                  min_samples_split = 5,
                                  max_features = 7,
                                  max_depth = 9,
                                  class_weight = {0:.35,1:.65},
                                  )

clf = clf.fit(X_train,y_train)
pred = clf.predict(X_test)

from sklearn.metrics import roc_auc_score
print "ROC AUC Score is ---- " + str(roc_auc_score(y_test,pred))
from sklearn.metrics import classification_report
print "Classification Report for Decision Tree Classifier ----------------------"
print classification_report(y_test,pred,target_names=['No Timeout','Timeout'])


"""
Use Grid Search to find the best parameters of the decision tree classifier to maximize the f1_score
"""
from sklearn.model_selection import GridSearchCV
clf_grid = tree.DecisionTreeClassifier()

parameters = {'criterion':['entropy','gini'],
              'splitter':['best'],
              'max_depth':[10,11,12],
              'min_samples_split':[5,10,15],
              'min_samples_leaf':[50,55,60],
              'max_features':[4,5,6],
              'min_impurity_split':[.08,.1,.12,.15],
              'class_weight':['balanced',{0:.3,1:.7},{0:.33,1:.67},{0:.28,1:.72}]
              }

clf_grid = GridSearchCV(clf_grid,parameters,scoring='roc_auc')
clf_grid.fit(X_train,y_train)
grid_params = clf_grid.best_params_

clf_grid = clf_grid.best_estimator_


print "Best GridSearch Parameters -------"
print grid_params

grid_pred = clf_grid.predict(X_test)
print "ROC AUC Score is ----" + str(roc_auc_score(y_test,grid_pred))
print "Classification Report for Decision Tree Classifier with Grid Search ----------------------"
print classification_report(y_test,grid_pred,target_names=['No Timeout','Timeout'])


"""
Export the Decision Tree Classifier decision graph in order to visualize the results. Text from the graph.txt file can
be input into http://webgraphviz.com/ in order to create visualization. There are some issues with installing the
graphviz library on Windows systems, which is why I decided to output to txt file rather than attempting to export
the visual to a pdf.
"""
import graphviz
graph_dot = tree.export_graphviz(clf,
                                 out_file = "graph.txt",
                                 feature_names=list(Def_X_df),
                                 class_names=['No Timeout','Timeout'],
                                 filled=True,
                                 rounded = True)
graph = graphviz.Source(graph_dot)

"""
Visualize the ROC curve of the best parameter with plotting

Sample code taken from http://scikit-learn.org/stable/auto_examples/model_selection/plot_roc_crossval.html#sphx-glr-auto-examples-model-selection-plot-roc-crossval-py
"""

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_curve, auc
from scipy import interp
import matplotlib.pyplot as plt


cv = StratifiedKFold(n_splits=15,random_state=15)
classifier = tree.DecisionTreeClassifier(criterion = 'entropy',
                                  splitter = 'best',
                                  min_impurity_split = .05,
                                  min_samples_leaf=8,
                                  min_samples_split = 5,
                                  max_features = 7,
                                  max_depth = 9,
                                  class_weight = {0:.35,1:.65},
                                  )

tprs = []
aucs = []
mean_fpr = np.linspace(0,1,100)

i = 0

for train, test in cv.split(Def_X_df,Def_y_df):

    clf = classifier.fit(Def_X_df.iloc[train],Def_y_df.iloc[train])
    probas_ = clf.predict_proba(Def_X_df.iloc[test])
    # Compute ROC curve and area under the curve
    fpr, tpr, thresholds = roc_curve(Def_y_df.iloc[test],probas_[:,1])
    tprs.append(interp(mean_fpr,fpr,tpr))
    tprs[-1][0] = 0.0
    roc_auc = auc(fpr,tpr)
    aucs.append(roc_auc)
    plt.plot(fpr, tpr, lw=1, alpha=0.3,
             #label = 'ROC fold %d (AUC = %0.2f)' % (i,roc_auc)
             )

    i += 1

plt.plot([0,1],[0,1], linestyle = '--', lw = 2, color = 'r', label = 'Luck', alpha = .8)

mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr,mean_tpr)
std_auc = np.std(aucs)
plt.plot(mean_fpr,mean_tpr, color= 'b', label = r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc,std_auc),
         lw=2, alpha=.8)

std_tpr = np.std(tprs, axis=0)
tprs_upper = np.minimum(mean_tpr + std_tpr,1)
tprs_lower = np.maximum(mean_tpr - std_tpr,0)
plt.fill_between(mean_fpr,tprs_lower,tprs_upper, color = 'grey', alpha = .2, label = r'$\pm$ 1 std. dev.')

plt.xlim([-0.05,1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Analysis of Decision Classifier')
plt.legend(loc = "lower right")
plt.show()

