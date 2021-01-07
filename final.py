#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 16 13:08:23 2020

@author: holdenbruce
"""

## Major Feedback from Part 1 That I Implemented in this Final Submission
# Improved Exploratory Data Analysis
    # - 3D plots
    # - more plots with better labels 
    # - more questions being asked of the data before diving in
# Implementing GridSearchCV 
    # - "while you don't *have* to use these capabilities [referring to GridSearchCV
    #   and KFold] for part 2 of the project, my personal feeling is that one coming 
    #   out of the ISLR level (but for Python) *absolutely SHOULD* have these tools 
    #   as their go to's" 
    # - I first implemented GridSearchCV for finding the optimal hyperparameters 
    #   through the use of param_grid
    # - Finally, I replaced the train_test_split with KFold, as Scott also 
    #   recommended I do this, arguing that it is a better method in every way
    # ... https://datascience.stackexchange.com/questions/52632/cross-validation-vs-train-validate-test
    # ... https://towardsdatascience.com/complete-guide-to-pythons-cross-validation-with-examples-a9676b5cac12
    # ... https://towardsdatascience.com/train-test-split-and-cross-validation-in-python-80b61beca4b6
    # ... https://towardsdatascience.com/5-reasons-why-you-should-use-cross-validation-in-your-data-science-project-8163311a1e79#:~:text=The%20training%20set%20is%20used,do%20more%20than%20one%20split.
    #   ^ this helped, use in conclusion / analysis in the comments for my submission
# Labeling Blue Tarps as 'positive' and not blue tarps as 'negative'
    # - in part 1 I had this designation flipped which, as Scott pointed out,
    #   is confusing since it goes against the convention of 1 (positive) being
    #   associated with the thing you are looking to find while 0 is what you 
    #   do not necessarily want. The change has been made for this final submission.
# Considered uncertainty of scores in the selection process.

# In the multiclass approach, it matters which of the classes I get wrong more than
# others...essentially arguing that some mistakes are worse than others.
    # - https://blog.tensorflow.org/2019/03/regression-with-probabilistic-layers-in.html



import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle

from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from scipy import interp
from sklearn.metrics import roc_auc_score

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn import neighbors
from sklearn.metrics import confusion_matrix, classification_report,accuracy_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split


from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RandomizedSearchCV

from sklearn.utils import resample # downsample the dataset
from sklearn import preprocessing # scale and center data
from sklearn.svm import SVC # this will make a support vector machine for classificaiton
from sklearn.metrics import plot_confusion_matrix # draws a confusion matrix
from sklearn.model_selection import GridSearchCV # this will do cross validation

# I will be using Plotly in this final submission of my project in an attempt
# to improve the readability and generally make the data more easily digestible
# to the common viewer.
# https://plotly.com/python/getting-started/
import plotly.express as px


# this sets the precision and general format for data output in this project
# one of the pieces of feedback i got from part 1 of this project was that 
# readability could be improved (primarily from me returning too many decimals
# so that it was making it hard to read tables), so i've chosen to specify the
# precision of decimals returned in pandas to 3
pd.set_option('precision', 3) # number precision for pandas
pd.set_option('display.max_rows', 12)
pd.set_option('display.max_columns', 12)
pd.set_option('display.float_format', '{:20,.4f}'.format) # get rid of scientific notation
plt.style.use('seaborn') # pretty matplotlib plots




pixels
holdout
## Load Data
pixels = pd.read_csv('HaitiPixels.csv', na_values=["?"])

pixels.head() #print first 5 
def load_data(dataset):
    
    #set X and y and then create train and test data from the dataset
    X = dataset.iloc[:,1:4] #X is columns: Red, Green, Blue
    y = dataset.Class  #Y is the Class column
    #this remapping turns Blue Tarp into classification=1 while every other choice 
    #is mapped to 0...thus making it a binary 
    
    #if y is still an object containing multiple categories, map those categories to 0s and 1s
    #otherwise, don't do anything so that the X,y split from this function can be
    #used for other purposes
    if y.dtype != 'int64':
        dataset.Class = dataset.Class.astype('category')
        y = y.map({'Blue Tarp':1, 'Rooftop':0,'Soil':0,'Various Non-Tarp':0, 'Vegetation':0})

        #rewrite the Class column in pixels with the new mapped version stored in y
        pixels['Class'] = y
    
    return X,y

X,y = load_data(pixels)




# pixels['hexcode'] = [{}]
# [pixels['Red'],pixels['Green'],pixels['Blue']]


# going to use webcolors to convert RGB to hexcode 
# conda install -c conda-forge webcolors
import webcolors


X['hex'] = X['Red']
X['hex'] = webcolors.rgb_to_hex((X['Red'],X['Green'],X['Blue']))

X['Red'][0]
X = X.drop(['hex'], axis=1)

classes = ['Red','Green','Blue']
# for row in range(len(X)):
for row in range(len(X)):
    red = 0
    green = 0
    blue = 0
    counter = 0
    for color in classes:
        counter+=1
        if color == 'Red':
            red = X[color][row]
        if color == 'Green':
            green = X[color][row]
        if color == 'Blue':
            blue = X[color][row]
        # if counter == 3:
            X['hex'][row] = webcolors.rgb_to_hex((red,green,blue))
            red = 0
            green = 0
            blue = 0
            counter = 0
X['hex']

from mpl_toolkits import mplot3d
ax = plt.axes(projection='3d')
ax.scatter(np.reshape(X['Red'], -1), np.reshape(X['Green'], -1), np.reshape(X['Blue'], -1), c=X['hex'])
ax.set_title('Red, Green, Blue Scattered')
ax.set_xlabel('Red')
ax.set_ylabel('Green')
ax.set_zlabel('Blue')
plt.show()



# Exploratory Data Analysis

# 3D plots

# https://yzhong-cs.medium.com/beyond-data-scientist-3d-plots-in-python-with-examples-2a8bd7aa654b
# from mpl_toolkits import mplot3d
# import matplotlib.pyplot as plt
# %matplotlib inline
plt.rcParams["figure.figsize"] = 12.8, 9.6

# ax = plt.axes(projection='3d')

# X1 = np.reshape(pixels['Red'], -1)
# Y1 = np.reshape(pixels['Green'], -1)
# Z1 = np.reshape(pixels['Blue'], -1)

# ax.plot_wireframe(X1,Y1,Z1)

# X5 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8]
# Y5 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
# Z5 = [21384, 29976, 15216, 4584, 10236, 7546, 6564, 2844, 4926, 7722, 4980, 2462, 12768, 9666, 2948, 6548, 10776, 8260, 8674, 5584, 5382, 10542, 12544, 5268, 25888, 31220, 9064, 7536, 6618, 1928, 9030, 5790, 6076, 8290, 8692, 4006, 14722, 11016, 2818, 9458, 3054, 5976, 1102, 1084, 9700, 8904, 12510, 11176, 10712, 6548, 2600, 5070, 6538, 4514, 1036, 292, 12572, 6534, 4478, 18500, 10452, 1912, 14254, 31050, 3880, 744, 990, 5534, 1670, 446, 2778, 8272, 14726, 27094, 872, 418, 884, 476, 2806, 1246, 1140, 922, 6202, 10848, 28828, 2360, 9660, 1412, 4296, 5272, 2854, 4150, 770, 5628, 4676, 3500, 31220, 10480, 5704, 5550, 1528, 3168, 2092, 2056, 1874, 7312, 938, 7428]
# x5 = np.reshape(X5, (9, 12))
# y5 = np.reshape(Y5, (9, 12))
# z5 = np.reshape(Z5, (9, 12))


# X5 = pd.DataFrame(X5)
# X5.unique()
# len(pixels['Red'].unique())


# from mpl_toolkits import mplot3d
# ax = plt.axes(projection='3d')
# ax.scatter(np.reshape(pixels['Red'], -1), np.reshape(pixels['Green'], -1), np.reshape(pixels['Blue'], -1), c=pixels['hexcode'])
# ax.set_title('Red, Green, Blue Scattered')
# ax.set_xlabel('Red')
# ax.set_ylabel('Green')
# ax.set_zlabel('Blue')
# plt.show()



sns.scatterplot(X.Red.values,X.Green.values,X.Blue.values)




    




#explore the shape of X and y
X.shape,y.shape #((63241, 3), (63241,))
pixels.head()

#rewrite the Class column in pixels with the new mapped version stored in y
# pixels['Class'] = y
# pixels.head() #looks good 
pixels.shape #(63241, 4)

#let's see what type of data is in each column
pixels.dtypes
# Class    category
# Red         int64
# Green       int64
# Blue        int64
# dtype: object

#alternatively
pixels.info() #get info on variables, looking for dtypes, could also use .dtypes

#check the variables 
X.Red.unique() #
X.Green.unique() #
X.Blue.unique() #
y.unique() #array([1, 0])



#







# In Part 1 of the project I used train-test-split to split the data into training data and 
# testing data. But for Part 2, I will use KFold to split the data.

## Test-Train Split
# def train_test(X,y):      
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42) #split 75% for train and 25% test
#     # X_train #A matrix containing the predictors associated with the training data (47,430 pixels)
#     # X_test #A matrix containing the predictors associated with the test data (15,811 pixels)
#     # y_train #A vector containing the class labels for the training observations, labeled Y_train below.
#     return X_train, X_test, y_train, y_test
# X_train, X_test, y_train, y_test = train_test(X,y)


# Now I will use kfold instead:

# Benefit #1 of KFold
# https://towardsdatascience.com/complete-guide-to-pythons-cross-validation-with-examples-a9676b5cac12
# Introducing cross-validation into the process helps you to reduce the need for the validation set because you’re able to train and test on the same data.
# ... maybe this will help with my ridiculously long run time for the SVM on the whole dataset?

# Benefit #2 of KFold
# Even though sklearn’s train_test_split method is using a stratified split, which means that the train and test set have the same distribution of the target variable, it’s possible that you accidentally train on a subset which doesn’t reflect the real world.

def kfold_train_test_split(X,y):

    #n_splits=10 means that the KFold will shift the test set 10 times
    #shuffle is False by default but if shuffle=True then splitting will be random
    kf = KFold(n_splits=10, shuffle=True, random_state=313)
    kf.get_n_splits(X)
    # print(kf)

    # for train_index, test_index in kf.split(X,y):
    #      # print("TRAIN:", np.take(X,train_index), "TEST:", test_index)
    #      X_train, X_test = X[train_index], X[test_index]
    #      y_train, y_test = y[train_index], y[test_index]
         
         # getting this error:
         # KeyError: "None of [Int64Index([    0,     1,     2,     3,     5,     6,     7,     8,     9,\n               10,\n            ...\n            63226, 63227, 63229, 63230, 63233, 63234, 63236, 63237, 63238,\n            63239],\n           dtype='int64', length=47430)] are in the [columns]"
         # realize that I cannot use a pandas dataframe since KFold uses numpy arrays
         # so need to convert the pandas dataframes i'm using into numpy arrays
         # in order for this to work
    
    # convert pandas dataframe into numpy array:
    X_np = X.to_numpy()
    y_np = y.to_numpy()
    
    # split the new X_np and y_np into train and test
    for train_index, test_index in kf.split(X_np):
         # print("TRAIN:", np.take(X_np,train_index), "TEST:", np.take(X_np, test_index))
          X_train, X_test = X_np[train_index], X_np[test_index]
          y_train, y_test = y_np[train_index], y_np[test_index]
         
         # now save the variables back as a pandas dataframe
         # X_train, X_test = pd.DataFrame(data=X_np[train_index], columns=['Red','Green','Blue']), pd.DataFrame(data=X_np[test_index], columns=['Red','Green','Blue'])
         # y_train, y_test = pd.DataFrame(data=y_np[train_index], columns=['Blue Tarp']), pd.DataFrame(data=y_np[test_index], columns=['Blue Tarp'])
    
    
    return X_train, X_test, y_train, y_test

X_train, X_test, y_train, y_test = kfold_train_test_split(X,y)





def scale_X_data(X_train_foo, X_test_foo):
    #then we will scale the data using StandardScaler 
    scaler = preprocessing.StandardScaler().fit(X_train_foo)
    X_train_foo_scaled = scaler.transform(X_train_foo)
    X_test_foo_scaled = scaler.transform(X_test_foo)
    
    return X_train_foo_scaled, X_test_foo_scaled

scale_X_data(X_train, X_test)

# X
# X_plot = X
# X
# X_plot.drop()
# X = X.drop(['hex'], axis=1)


### Methods

## KNN
# http://www.science.smith.edu/~jcrouser/SDS293/labs/lab3-py.html

#KNN-specific
def optimal_K(X_train, X_test, y_train, y_test):
    ### Testing for best/optimal K value
    # def find_optimal_k
    accuracy_rate_dict = {}
    for i in range(1,40):
        knn = neighbors.KNeighborsClassifier(n_neighbors=i)
        knn_pred = knn.fit(X_train, y_train).predict(X_test)
        accuracy_rate_dict[accuracy_score(y_test, knn_pred)] = i
    max(accuracy_rate_dict)
    best_k = accuracy_rate_dict[max(accuracy_rate_dict)] #20
    print('KNN performs best when the classifier k={}'.format(best_k))
    
    K = best_k #A value for K, the number of nearest neighbors to be used by the classifier.
    return K
 

#KNN-specific
def knn(X_train, X_test, y_train, y_test):   
    """This function works rather differently from the other model-fitting 
    functions that we have encountered thus far. Rather than a two-step approach 
    in which we first fit the model and then we use the model to make predictions,
    knn() forms predictions using a single command. The function requires four 
    inputs."""
    
    K = optimal_K(X_train, X_test, y_train, y_test)
        
    """Now the neighbors.KNeighborsClassifier() function can be used to predict the 
    classification of Blue Tarp, Rooftop, Soil, Various Non-Tarp, or Vegetation"""
    # fit a model
    knn = neighbors.KNeighborsClassifier(n_neighbors = K)
    
    #predict classification
    knn_pred = knn.fit(X_train, y_train).predict(X_test) #fit the model and predict
    
    # predict probabilities
    knn_probs = knn.fit(X_train, y_train).predict_proba(X_test) #fit the model and predict
    
    # keep probabilities for the positive outcome only
    knn_probs = knn_probs[:, 1]
    
    return knn_pred, knn_probs
knn_pred,knn_probs = knn(X_train, X_test, y_train, y_test)




## LDA -- Linear Discriminant Analysis
def lda(X_train, X_test, y_train, y_test):
    # fit a model
    lda = LinearDiscriminantAnalysis()
    
    #predict classification
    lda_pred = lda.fit(X_train, y_train).predict(X_test)
    
    # predict probabilities
    lda_probs = lda.fit(X_train, y_train).predict_proba(X_test)
    
    # keep probabilities for the positive outcome only
    lda_probs = lda_probs[:, 1]

    return lda_pred, lda_probs
lda_pred, lda_probs = lda(X_train, X_test, y_train, y_test)




## QDA -- Quadratic Discriminant Analysis
def qda(X_train, X_test, y_train, y_test):
    # fit a model
    qda = QuadraticDiscriminantAnalysis()
    
    #predict classification
    qda_pred = qda.fit(X_train, y_train).predict(X_test)
    
    # predict probabilities
    qda_probs = qda.fit(X_train, y_train).predict_proba(X_test)
    
    # keep probabilities for the positive outcome only
    qda_probs = qda_probs[:, 1]

    return qda_pred, qda_probs
qda_pred, qda_probs = qda(X_train, X_test, y_train, y_test)



## Logistic Regression
def logistic_regression(X_train, X_test, y_train, y_test):
    # fit a model
    model = LogisticRegression(solver='lbfgs')
    
    #predict classification
    logreg_pred = model.fit(X_train, y_train).predict(X_test)
    
    # predict probabilities
    logreg_probs = model.fit(X_train, y_train).predict_proba(X_test)
    
    # keep probabilities for the positive outcome only
    logreg_probs = logreg_probs[:, 1]

    return logreg_pred, logreg_probs
logreg_pred, logreg_probs = logistic_regression(X_train, X_test, y_train, y_test)






# Part 2

# Now we can create the random forest model
## Bagging:
def bagging(X_train, X_test, y_train, y_test):
    # fit a model
    rfc = RandomForestClassifier(max_features=X_train.shape[1],random_state=313)
    
    #predict classification
    bagging_pred = rfc.fit(X_train, y_train).predict(X_test)
    
    # predict probabilities
    bagging_probs = rfc.fit(X_train, y_train).predict_proba(X_test)
    
    # keep probabilities for the positive outcome only
    bagging_probs = bagging_probs[:, 1]

    return bagging_pred, bagging_probs, rfc
bagging_pred, bagging_probs, rfc = bagging(X_train, X_test, y_train, y_test)


# Let’s next evaluate how the model performed.

# Evaluating Performance:
    
def eval_perform_rf(rfc, X, y, y_test, bagging_pred):
    bagging_cv_score = cross_val_score(rfc, X, y, cv=10, scoring='roc_auc', n_jobs=-1)
    # bagging_cv_score = cross_val_score(rfc, X, y, cv=10, scoring='roc_auc') #without specifying n_jobs
    for i in range(len(bagging_cv_score)):
        print('CV={}: {}'.format(i+1,bagging_cv_score[i]))
    print('\n')
    
    print("=== Confusion Matrix ===")
    print(confusion_matrix(y_test, bagging_pred))
    print('\n')
    
    print("=== Classification Report ===")
    print(classification_report(y_test, bagging_pred))
    print('\n')
    
    print("=== All AUC Scores ===")
    print(bagging_cv_score)
    print('\n')
    
    print("=== Mean AUC Score ===")
    print("Mean AUC Score - Random Forest: ", bagging_cv_score.mean())
    
eval_perform_rf(rfc, X, y, y_test, bagging_pred)

# CV=1: 1.0
# CV=2: 0.9999482470298647
# CV=3: 0.9975247524752475
# CV=4: 1.0
# CV=5: 1.0
# CV=6: 1.0
# CV=7: 0.9868721313490382
# CV=8: 0.9940132325875515
# CV=9: 0.9081534378527693
# CV=10: 0.9993541574954349


# === Confusion Matrix ===
# [[6123    4]
#  [   7  190]]


# === Classification Report ===
#               precision    recall  f1-score   support

#            0       1.00      1.00      1.00      6127
#            1       0.98      0.96      0.97       197

#     accuracy                           1.00      6324
#    macro avg       0.99      0.98      0.99      6324
# weighted avg       1.00      1.00      1.00      6324



# === All AUC Scores ===
# [1.         0.99994825 0.99752475 1.         1.         1.
#  0.98687213 0.99401323 0.90815344 0.99935416]


# === Mean AUC Score ===
# Mean AUC Score - Random Forest:  0.9885865958789907


# performs very well: 98.86% accuracy - AUC Score 




    
    
    
    
# The confusion matrix is useful for giving you false positives and false negatives. 
# The classification report tells you the accuracy of your model. 
# The ROC curve plots out the true positive rate versus the false positive rate at various thresholds. 
# The roc_auc scoring used in the cross-validation model shows the area under the ROC curve.

# We’ll evaluate our model’s score based on the roc_auc score (stored in bagging_cv_score.mean()
# and returned as "Mean AUC Score - Rnadom Forest"), which is 0.9886

# The next thing we should do is tune our hyperparameters to see if we can improve 
# the performance of the model.
# https://towardsdatascience.com/hyperparameter-tuning-the-random-forest-in-python-using-scikit-learn-28d2aa77dd74

# Tuning Hyperparameters
# I’d recommend William Koehrsen’s article, “Hyperparameter Tuning the Random Forest in Python” for a more detailed description of the process. We’ll do a cliff-notes version.
# We’ll use RandomizedSearchCV from sklearn to optimize our hyperparamaters. 
# Koehrsen uses a full grid of hyperparameters in his article, but I found that this could take a very substantial time to run in practice. I decided to focus on 3 hyperparameters: n_estimators, max_features, and max_depth.

def rf_tune_hyperparameters(rfc, X_train, y_train):
    #n_estimators determines the number of trees in the random forest
    #take 11 values of n_estimators starting from 100 and ending with 2000, equally spaced 
    # n_estimators = [int(x) for x in np.linspace(start = 100, stop = 2000, num = 11)] #default is 100
    n_estimators = [10, 50, 100, 500, 1000]
    
    
    # number of features at every split
    max_features = ['auto', 'sqrt','log2', np.random.randint(1,4)] #default is auto in sklearn 
    
    # max_depth determines the maximum depth of the tree
    #take 11 values of max_depth starting from 100 and ending with 500, equally spaced 
    max_depth = [int(x) for x in np.linspace(100, 500, num = 11)]
    # max_depth = [2,3,4,5,6,7,8,9, 10, None] #None is the default for max_depth in sklearn 
    # max_depth = [2,3,4]
    # criterion = ['gini','entropy']
    
    # create grid of these parameters that will be passed into the searchCV function
    param_grid = {
     'n_estimators': n_estimators,
     'max_features': max_features,
     'max_depth': max_depth
     }
    
    
       
    rfc_random.fit(X_train.to_numpy(), y_train.to_numpy())
    # this is a pretty data intensive model fit which is why i have set n_jobs to -1
    #in the RandomizedSearchCV(), which means that all of my processors are being 
    #used in parallel to run this fit

    # print results
    # print(rfc_random.best_params_)
    
    #now store the optimal parameters and then return them for use in other functions
    optimal_n_estimators = rfc_random.best_params_['n_estimators']
    optimal_max_features = rfc_random.best_params_['max_features']
    optimal_max_depth = rfc_random.best_params_['max_depth']
    # optimal_criterion = rfc_random.best_params_['criterion']
    

    
    print("=== Optimal n_estimators ===")
    print(optimal_n_estimators)
    print('\n')
    
    print("=== Optimal max_features ===")
    print(optimal_max_features)
    print('\n')
    
    print("=== Optimal max_depth ===")
    print(optimal_max_depth)
    print('\n')
    
    return rfc_random, optimal_n_estimators, optimal_max_features, optimal_max_depth

rfc_random, optimal_n_estimators, optimal_max_features, optimal_max_depth = rf_tune_hyperparameters(rfc, X_train, y_train)



# now we can plug these back into the model to see if it improved our performance
def rf_optimal():
    rfc_optimal = RandomForestClassifier(n_estimators=optimal_n_estimators, max_depth=optimal_max_depth, max_features=optimal_max_features, random_state=313, n_jobs=-1)
    rfc_optimal.fit(X_train,y_train)
    rfc_optimal_predict = rfc_optimal.predict(X_test)
    rfc_optimal_cv_score = cross_val_score(rfc_optimal, X, y, cv=10, scoring='roc_auc', n_jobs=-1)
    for i in range(len(rfc_optimal_cv_score)):
        print('CV={}: {}'.format(i+1,rfc_optimal_cv_score[i]))
    print('\n')        
    
    print("=== Confusion Matrix ===")
    print(confusion_matrix(y_test, rfc_optimal_predict))
    print('\n')
    # === Confusion Matrix ===
    # [[15289    18]
    #  [   25   479]]
    
    print("=== Classification Report ===")
    print(classification_report(y_test, rfc_optimal_predict))
    print('\n')
    
    print("=== All AUC Scores ===")
    print(rfc_optimal_cv_score)
    print('\n')
    
    print("=== Mean AUC Score ===")
    print("Mean AUC Score - Random Forest: ", rfc_optimal_cv_score.mean())
    
    return rfc_optimal_cv_score

rf_opimal_cv_score = rf_optimal()

# CV=1: 1.0
# CV=2: 0.9999142841432134
# CV=3: 0.9975247524752475
# CV=4: 1.0
# CV=5: 1.0
# CV=6: 1.0
# CV=7: 0.9993021435433318
# CV=8: 0.9992443257720087
# CV=9: 0.8978145691080053
# CV=10: 0.9992322320880309


# === Confusion Matrix ===
# [[6121    6]
#  [   7  190]]


# === Classification Report ===
#               precision    recall  f1-score   support

#            0       1.00      1.00      1.00      6127
#            1       0.97      0.96      0.97       197

#     accuracy                           1.00      6324
#    macro avg       0.98      0.98      0.98      6324
# weighted avg       1.00      1.00      1.00      6324



# === All AUC Scores ===
# [1.         0.99991428 0.99752475 1.         1.         1.
#  0.99930214 0.99924433 0.89781457 0.99923223]


# === Mean AUC Score ===
# Mean AUC Score - Random Forest:  0.9893032307129838

# Also performs incredibly well: 98.93% accuracy - AUC score
# This is slightly higher than bagging 
# But the confusion matrix actually shows that 2 observations that had been
# correctly classified ended up being incorrectly classified 






















### Support Vector Machines

### SVM

# Now we can create the SVM model for filling out the table
## Bagging:
def svm(X_train, X_test, y_train, y_test):
    # fit a model
    clf_svm = SVC(random_state=313, probability=True)
    clf_svm.fit(X_train, y_train)
        
    #predict classification
    clf_svm_pred = clf_svm.fit(X_train, y_train).predict(X_test)
    
    # predict probabilities
    clf_svm_probs = clf_svm.fit(X_train, y_train).predict_proba(X_test)
    #AttributeError: predict_proba is not available when  probability=False
    
    # keep probabilities for the positive outcome only
    clf_svm_probs = clf_svm_probs[:, 1]

    return clf_svm_pred, clf_svm_probs, clf_svm
clf_svm_pred, clf_svm_probs, clf_svm = svm(X_train, X_test, y_train, y_test)


len(pixels) #63241
pixels.head()

## Downsample the data 
# 63241 samples will take a long time for the support vector machine to run
# so downsampling will make this process quicker...
# Support Vector Machines are great with small datasets, but not awesome with large ones, 
# and this dataset is big enough to take a long time to optimize with Cross Validation.
# So we'll downsample both categories, Blue Tarp and Non Blue Tarp to 1,000 each.
#     - rationale taken from statquest support vector machine video

def downsample_data(num_samples):    
    #to ensure that i actually get 1000 samples for each category, i'll start by 
    #splitting the data into two dataframes, one for Class=1 (blue tarp) and one 
    #for Class=0 (not blue tarp)
    pixels_blue_tarp = pixels[y==1] #pixels[pixels['Class']==1]
    pixels_not_blue_tarp = pixels[y==0] # pixels[pixels['Class']==0]
    
    #let's see how this breaks down
    pixels_blue_tarp.shape #(2022, 4) ... 2022 observations 
    pixels_not_blue_tarp.shape #(61219, 4) ... 61219 observations
    sum(y)/len(y) #0.03197292895431761
    #definitely an unbalanced dataset where there are way more examples of non-blue tarp
    #observations than there are blue tarp observations (this is expected)
    
    #now downsample the pixels_blue_tarp dataframe 
    pixels_blue_tarp_downsampled = resample(pixels_blue_tarp,
                                            replace=False,
                                            n_samples=num_samples, #pass in num_samples
                                            random_state=313)
    
    #now downsample the pixels_not_blue_tarp dataframe
    pixels_not_blue_tarp_downsampled = resample(pixels_not_blue_tarp,
                                            replace=False,
                                            n_samples=num_samples, #pass in num_samples
                                            random_state=313)
    
    #now merge the two downsampled dataframes back into a single dataframe 
    pixels_downsampled = pd.concat([pixels_blue_tarp_downsampled, pixels_not_blue_tarp_downsampled])
    #then print out the total number of samples (should be 2000 or 2*num_samples) to make sure i did this correctly
    len(pixels_downsampled) #2000

    return pixels_downsampled

pixels_downsampled = downsample_data(num_samples=1000)

pixels_downsampled.head()
pixels_downsampled.shape




# now that we have formatted our downsampled data, we can start building the support 
# vector machine ... first, we split the data into two parts: the column of data we  
# will use to make classifications and the column of data that we want to predict

#split this data using the load_data() from above and pass it the downsampled data
X_downsampled, y_downsampled = load_data(pixels_downsampled)
#nothing to one hot encode in X

#make sure it looks good
X_downsampled.shape
X_downsampled.head()
y_downsampled.shape
y_downsampled.head()


# but the radial basis function that we use with our support vector machine assumes
# that the data we feed it are centered and scaled, so we need to make sure that 
# each column should have a mean value=0 and a standard deviation=1 for both the 
# training and testing dataset

# We’ll use the train_test() function we wrote earlier to split the data into 
# training data and testing data
X_train_downsampled, X_test_downsampled, y_train_downsampled, y_test_downsampled = kfold_train_test_split(X_downsampled, y_downsampled)

#then we will scale the data using StandardScaler 
scaler = preprocessing.StandardScaler().fit(X_train_downsampled)
X_train_downsampled_scaled = scaler.transform(X_train_downsampled)
X_test_downsampled_scaled = scaler.transform(X_test_downsampled)








# this function builds the SVM model and can take in either downsampled data or otherwise 
# but because of SVM's structure, it expects scaled data 
def build_svm(X_train_foo_scaled, X_test_foo_scaled, y_train_foo, y_test_foo, clf_svm = SVC(random_state=313)):

    #now that we have our data split into test and train we can build our preliminary 
    #support vector machine
    # clf_svm = SVC(random_state=313)
    clf_svm.fit(X_train_foo_scaled, y_train_foo)

    #that's it, that's our support vector machine for classification
    #now we can draw a confusion matrix and see how it performs on the test data 

    plot_confusion_matrix(clf_svm, 
                      X_test_foo_scaled, 
                      y_test_foo,
                      values_format='d',
                      display_labels=["Not Blue Tarp", "Blue Tarp"])

build_svm(X_train_downsampled_scaled,X_test_downsampled_scaled,y_train_downsampled, y_test_downsampled)

# this performs incredibly well 
# ...
# might i be overfitting? these tests are performing so well
# or is this just evidence of random forest and SVMs being better solutions for 
# this kind of classification problem than the methods used in part 1?
# ...
# the confusmion matrix shows us that of the 264 observations that were 
# not blue tarp, 258 (97.7%) were correctly classified
# of the 236 observatiosn that were blue tarps, 235 (99.6%) were correctly
# classified 
# this goes to show how great SVM is as an out-of-the-box solution / machine
# learning technique...
# i will use cross validation to optimize the parameters to see if i can improve
# predictions at all but honestly, these numbers are already fantastic...my next
# step would be to run the entire dataset (without downsampling) to see how it
# performs 










def optimizing_svm():
    # when we are optimizing a support vector machine we are attempting to find the 
    # best value for gamma and the regularization parameter C that will improve the
    # prediction/classification accuracy of the model 
    # since we are looking to optimize two parameters (gamma and C) we will use 
    # GridSearchCV(), specifying several values for gamma and C and then letting
    # the function determine the optimal combination (that's the great thing about using
    # GridSerachCV() for problems like this: it tests all possible combinations of 
    # parameters for us and all we have to do is plug in some options)
    
    # the default values for the SVC parameters in sklearn.svm are:
    # C=1.0
    # kernel='rbf
    # degree=3
    # gamma='scale' (which is 1/(n_features*X.var()) while 'auto' uses 1/n_features
    
    # GridSearchSV in sklearn.model_selection has a parameter called param_grid
    # where you can pass in a dictionary of parameters 
    # i will pass in all of the default SVC parameters and then some extras
    # so that the GridSearchCV() can find the optimal combination from many options
    
    # i might not need to have all these options for kernel because my assumption
    # is that 'rbf' will work the best for the pixel data, but i am including 
    # 'linear','poly', and 'sigmoid' 
    param_grid = [
      {'C': np.arange(0.001,5.001,.1), 
       'degree':[1,2,3,4],
       'gamma': ['scale', 'auto', 1, 0.1, 0.01, 0.001, 0.0001], 
       'kernel': ['rbf','linear','poly','sigmoid']},
    ]
    
    optimal_params = GridSearchCV(
        SVC(),
        param_grid,
        cv=5,
        scoring='accuracy',
        n_jobs=-1)

    return optimal_params

optimal_params = optimizing_svm()



#now that we have the optimal parameters for the SVM model, we need to fit the 
# model with the optimal parameter grid ... searching through best options
optimal_params.fit(X_train_downsampled_scaled, y_train_downsampled)
#takes a little while 
# GridSearchCV(cv=5, estimator=SVC(), n_jobs=-1,
#              param_grid=[{'C': array([1.000e-03, 1.010e-01, 2.010e-01, 3.010e-01, 4.010e-01, 5.010e-01,
#        6.010e-01, 7.010e-01, 8.010e-01, 9.010e-01, 1.001e+00, 1.101e+00,
#        1.201e+00, 1.301e+00, 1.401e+00, 1.501e+00, 1.601e+00, 1.701e+00,
#        1.801e+00, 1.901e+00, 2.001e+00, 2.101e+00, 2.201e+00, 2.301e+00,
#        2.401e+00, 2.501e+00, 2.601e+00, 2.701e+00, 2.801e+00, 2.901e+00,
#        3.001e+00, 3.101e+00, 3.201e+00, 3.301e+00, 3.401e+00, 3.501e+00,
#        3.601e+00, 3.701e+00, 3.801e+00, 3.901e+00, 4.001e+00, 4.101e+00,
#        4.201e+00, 4.301e+00, 4.401e+00, 4.501e+00, 4.601e+00, 4.701e+00,
#        4.801e+00, 4.901e+00]),
#                           'degree': [1, 2, 3, 4],
#                           'gamma': ['scale', 'auto', 1, 0.1, 0.01, 0.001,
#                                     0.0001],
#                           'kernel': ['rbf', 'linear', 'poly', 'sigmoid']}],
#              scoring='accuracy')


#save the optimal parameters down in their own varaibles

best_C = optimal_params.best_params_['C']
best_degree = optimal_params.best_params_['degree']
best_gamma = optimal_params.best_params_['gamma']
best_kernel = optimal_params.best_params_['kernel']

#now print out the optimal parameters for the model
print('''The best cost for clf_svm is: {}
      the best degree for clf_svm is: {}
      the best gamma for clf_svm is: {}
      the best kernel for clf_svm is: '{}'
      '''.format(
      best_C,
      best_degree,
      best_gamma,
      best_kernel
      # optimal_params.best_params_['C'],
      # optimal_params.best_params_['degree'],
      # optimal_params.best_params_['gamma'],
      # optimal_params.best_params_['kernel']
)) 
# The best cost for clf_svm is: 4.7010000000000005
#       the best degree for clf_svm is: 1
#       the best gamma for clf_svm is: 1
#       the best kernel for clf_svm is: 'rbf'


    
#then use those optimal parameters to fit a new model 
# clf_svm_optimal = SVC(random_state=313, C=0.901, degree=1, gamma=1, kernel='rbf')
clf_svm_optimal = SVC(random_state=313, C=best_C, degree=best_degree, gamma=best_gamma, kernel=best_kernel)
build_svm(X_train_downsampled_scaled, X_test_downsampled_scaled, y_train_downsampled, y_test_downsampled, clf_svm = clf_svm_optimal)

# clf_svm_optimal.fit(X_train_downsampled_scaled, y_train_downsampled)
#now i'll draw a confusion matrix for this optimal svc model to see if 
#tuning the parameters improved the support vector machine any more than
#the original fit, which was already very good
# plot_confusion_matrix(clf_svm_optimal, 
#                       X_test_downsampled_scaled, 
#                       y_test_downsampled,
#                       values_format='d',
#                       display_labels=["Not Blue Tarp", "Blue Tarp"])
# #this performs identically (incredibly) well 
# ... 
#the confusmion matrix shows us that of the 264 observations that were 
#not blue tarp, 258 (97.7%) were correctly classified
#of the 236 observatiosn that were blue tarps, 235 (99.6%) were correctly
#classified 
#this goes to show how great SVM is as an out-of-the-box solution / machine
#learning technique...despite using cross validation to optimize the parameters 
#to improve predictions, these numbers were already optimal from the start





###############

## ok this is the same as before
# i'm becoming skeptical of the reliability of downsampling
# maybe i'm using too few observations?
# i'll try downsampling with more data points, maybe 5000?

pixels_downsampled = downsample_data(num_samples=5000)
#ValueError: Cannot sample 5000 out of arrays with dim 2022 when replace is False

#ah, but there are only 2022 observations where the pixel represents a blue tarp
#so downsampling to 5000 from this dataset is not an option
#but maybe this could be an option for the larger 2M holdout dataset we were 
#provided for part 2 of this project? that might be a more appropriate usecase for
#testing the effectivness of downsampling 



pixels_downsampled = downsample_data(num_samples=2022)

pixels_downsampled.head()
pixels_downsampled.shape

#now that we have formatted our downsampled data, we can start building the support 
#vector machine ... first, we split the data into two parts: the column of data we  
#will use to make classifications and the column of data that we want to predict

#split this data using the load_data() from above and pass it the downsampled data
X_downsampled, y_downsampled = load_data(pixels_downsampled)
#nothing to one hot encode in X

#make sure it looks good
X_downsampled.shape
X_downsampled.head()
y_downsampled.shape
y_downsampled.head()


#but the radial basis function that we use with our support vector machine assumes
#that the data we feed it are centered and scaled, so we need to make sure that 
#each column should have a mean value=0 and a standard deviation=1 for both the 
#training and testing dataset

# We’ll use the train_test() function we wrote earlier to split the data into 
# training data and testing data
X_train_downsampled, X_test_downsampled, y_train_downsampled, y_test_downsampled = kfold_train_test_split(X_downsampled, y_downsampled)

#then we will scale the data using StandardScaler 
scaler = preprocessing.StandardScaler().fit(X_train_downsampled)
X_train_downsampled_scaled = scaler.transform(X_train_downsampled)
X_test_downsampled_scaled = scaler.transform(X_test_downsampled)

build_svm(X_train_downsampled_scaled,X_test_downsampled_scaled,y_train_downsampled, y_test_downsampled)
optimal_params = optimizing_svm()
optimal_params.fit(X_train_downsampled_scaled, y_train_downsampled)

best_C = optimal_params.best_params_['C']
best_degree = optimal_params.best_params_['degree']
best_gamma = optimal_params.best_params_['gamma']
best_kernel = optimal_params.best_params_['kernel']

#now print out the optimal parameters for the model
print('''The best cost for clf_svm is: {}
      the best degree for clf_svm is: {}
      the best gamma for clf_svm is: {}
      the best kernel for clf_svm is: '{}'
      '''.format(
      best_C,
      best_degree,
      best_gamma,
      best_kernel
      # optimal_params.best_params_['C'],
      # optimal_params.best_params_['degree'],
      # optimal_params.best_params_['gamma'],
      # optimal_params.best_params_['kernel']
)) 
    
#then use those optimal parameters to fit a new model 
# clf_svm_optimal = SVC(random_state=313, C=0.901, degree=1, gamma=1, kernel='rbf')
clf_svm_optimal = SVC(random_state=313, C=best_C, degree=best_degree, gamma=best_gamma, kernel=best_kernel)
clf_svm_optimal.fit(X_train_downsampled_scaled, y_train_downsampled)

#now i'll draw a confusion matrix for this optimal svc model to see if 
#tuning the parameters improved the support vector machine any more than
#the original fit, which was already very good
plot_confusion_matrix(clf_svm_optimal, 
                      X_test_downsampled_scaled, 
                      y_test_downsampled,
                      values_format='d',
                      display_labels=["Not Blue Tarp", "Blue Tarp"])


## this is nearly perfect classification!
###############









#the next step would still be to run the entire dataset (without downsampling) 
#to see how it performs 

pixels.info()
pixels.head()
X,y

# We’ll use the train_test() function we wrote earlier to split the data into 
# training data and testing data
X_train_full, X_test_full, y_train_full, y_test_full = kfold_train_test_split(X, y)

#then we will scale the data using StandardScaler 
# scaler = preprocessing.StandardScaler().fit(X_train_full)
# X_train_full_scaled = scaler.transform(X_train_full)
# X_test_full_scaled = scaler.transform(X_test_full)
X_train_full_scaled, X_test_full_scaled = scale_X_data(X_train_full, X_test_full)



#now that we have our data split into test and train we can build our preliminary 
#support vector machine

build_svm(X_train_full_scaled, X_test_full_scaled, y_train_full, y_test_full)


#with optimal parameters:
clf_svm_optimal = SVC(random_state=313, C=best_C, degree=best_degree, gamma=best_gamma, kernel=best_kernel)
build_svm(X_train_full_scaled, X_test_full_scaled, y_train_full, y_test_full, clf_svm = clf_svm_optimal)




# clf_svm = SVC(random_state=313)
# clf_svm.fit(X_train_full_scaled, y_train_full)

# #that's it, that's our support vector machine for classification
# #now we can draw a confusion matrix and see how it performs on the test data 

# plot_confusion_matrix(clf_svm, 
#                       X_test_full_scaled, 
#                       y_test_full,
#                       values_format='d',
#                       display_labels=["Not Blue Tarp", "Blue Tarp"])
















############################  Completing Table 1  ############################

## Accuracy
#can be used for KNN, LDA, QDA, Logistic Regression
def accuracy(y_test, pred, method):
    accuracy = accuracy_score(y_test, pred)
    print('{} Accuracy: {}'.format(method,str(accuracy)))
    print('{} Test Error: {}'.format(method, str(1 - accuracy )))
    """that's actually not very helpful because this is scoring across classes, which 
    isn't giving us a good view of the accuracy of the model...use this instead:"""
    
    return accuracy
    # ^^ there is a severe imbalance
    #https://machinelearningmastery.com/threshold-moving-for-imbalanced-classification/
    #need to move threshold to accommodate this imbalance 
    #"It has been stated that trying other methods, such as sampling, without trying by simply setting the threshold may be misleading. The threshold-moving method uses the original training set to train [a model] and then moves the decision threshold such that the minority class examples are easier to be predicted correctly."
    #Pages 72, Imbalanced Learning: Foundations, Algorithms, and Applications, 2013.

knn_accuracy = accuracy(y_test,knn_pred,'KNN')
lda_accuracy = accuracy(y_test,lda_pred,'LDA')
qda_accuracy = accuracy(y_test,qda_pred,'QDA')
logreg_accuracy = accuracy(y_test,logreg_pred,'Logistic Regression')
bagging_accuracy = accuracy(y_test, bagging_pred,'Bagging')
svm_accuracy = accuracy(y_test, clf_svm_pred, 'SVM')




### SVM

def svm(X_train_svm, X_test_svm, y_train_svm, y_test_svm):
    # fit a model
    clf_svm = SVC(random_state=313, probability=True)
    clf_svm.fit(X_train_svm, y_train_svm)
        
    #predict classification
    clf_svm_pred = clf_svm.fit(X_train_svm, y_train_svm).predict(X_test_svm)
    
    # predict probabilities
    clf_svm_probs = clf_svm.fit(X_train_svm, y_train_svm).predict_proba(X_test_svm)
    #AttributeError: predict_proba is not available when  probability=False
    
    # keep probabilities for the positive outcome only
    clf_svm_probs = clf_svm_probs[:, 1]

    return clf_svm_pred, clf_svm_probs, clf_svm





# This function builds the SVM model and can take in either downsampled data or otherwise 
# but because of SVM's structure, it expects scaled data 
def build_svm(X_train_foo_scaled, X_test_foo_scaled, y_train_foo, y_test_foo, clf_svm = SVC(random_state=313)):
    # Now that we have our data split into test and train we can build our preliminary 
    # support vector machine
    
    # clf_svm = SVC(random_state=313)
    clf_svm.fit(X_train_foo_scaled, y_train_foo)

    #that's it, that's our support vector machine for classification
    #now we can draw a confusion matrix and see how it performs on the test data 

    plot_confusion_matrix(clf_svm, 
                      X_test_foo_scaled, 
                      y_test_foo,
                      values_format='d',
                      display_labels=["Not Blue Tarp", "Blue Tarp"])
    
    
    
# This function builds the SVM model and can take in either downsampled data or otherwise 
# but because of SVM's structure, it expects scaled data 
def build_svm(X_train_foo_scaled, X_test_foo_scaled, y_train_foo, y_test_foo, clf_svm = SVC(random_state=313, probability=True, verbose=False)):
    # Now that we have our data split into test and train we can build our preliminary 
    # support vector machine
        
    # clf_svm = SVC(random_state=313)
    clf_svm.fit(X_train_foo_scaled, y_train_foo)
    
    clf_svm.verbose = False 
    
    #predict classification
    clf_svm_pred = clf_svm.fit(X_train_foo_scaled, y_train_foo).predict(X_test_foo_scaled)
    
    # predict probabilities
    clf_svm_probs = clf_svm.fit(X_train_foo_scaled, y_train_foo).predict_proba(X_test_foo_scaled)
    #AttributeError: predict_proba is not available when  probability=False
    
    # keep probabilities for the positive outcome only
    clf_svm_probs = clf_svm_probs[:, 1]

    #that's it, that's our support vector machine for classification
    #now we can draw a confusion matrix and see how it performs on the test data 

    plot_confusion_matrix(clf_svm, 
                      X_test_foo_scaled, 
                      y_test_foo,
                      values_format='d',
                      display_labels=["Not Blue Tarp", "Blue Tarp"])
    
    
    return clf_svm_pred, clf_svm_probs, clf_svm
    
    
    



## AUC 

def calculate_AUC(y_test, prob):
    # We can then use the roc_auc_score() function to calculate the true-positive rate and 
    #false-positive rate for the predictions using a set of thresholds that can then be used 
    #to create a ROC Curve plot.
    
    # calculate scores
    auc = roc_auc_score(y_test, prob)
    return auc
auc_KNN = calculate_AUC(y_test, knn_pred)  
print(auc_KNN)
auc_LDA = calculate_AUC(y_test, lda_pred)  
print(auc_LDA)
auc_QDA = calculate_AUC(y_test, qda_pred)  
print(auc_QDA)
auc_LogisticRegression = calculate_AUC(y_test, logreg_pred) 
print(auc_LogisticRegression)
auc_Bagging = calculate_AUC(y_test, bagging_pred) 
print(auc_Bagging)
auc_SVM = calculate_AUC(y_test, clf_svm_pred) 
print(auc_SVM)








## Threshold for ROC 

def best_threshold(fpr, tpr, thresholds):
    # Youden's J-statistic for calculating optimal threshold
    # https://en.wikipedia.org/wiki/Youden%27s_J_statistic
    J = tpr - fpr 
    
    best_index = 0
    #loop through J and set 'best_index' to whichever corresponds to the max(J)
    for index,value in enumerate(J):
        if value==max(J):
            best_index=index
            # print(best_index)

    best_thresh = thresholds[best_index]    
    print('Best Threshold={:f}'.format(best_thresh))   
    return best_index, best_thresh




## ROC 

def calculate_ROC(y_test, prob, Type):    
    # calculate roc curves
    fpr, tpr, thresholds = roc_curve(y_test, prob)
    
    best_index, best_thresh = best_threshold(fpr, tpr, thresholds)
    # plot the roc curve for the model
    plt.plot([0,1], [0,1], linestyle='--', label='No Skill')
    plt.plot(fpr, tpr, marker='.', label='{} ROC'.format(Type))
    #best threshold
    plt.scatter(fpr[best_index], tpr[best_index], marker='o', color='black', label='Optimal Threshold = {:3f}'.format(best_thresh))
    # axis labels
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    # show the plot
    plt.show()


    
roc_KNN = calculate_ROC(y_test, knn_probs, Type='K-Nearest Neighbors')  
roc_LDA = calculate_ROC(y_test, lda_probs, Type='LDA')  
roc_QDA = calculate_ROC(y_test, qda_probs, Type='QDA')  
roc_LogisticRegression = calculate_ROC(y_test, logreg_probs,Type='Logistic Regression')  
roc_Bagging = calculate_ROC(y_test, bagging_probs,Type='Bagging')  
roc_SVM = calculate_ROC(y_test, clf_svm_probs,Type='SVM')  








## Confusion Matrix 

#can be used for KNN, LDA, QDA, Logistic Regression
def conf_m(y_test,pred):    
    """The confusion_matrix() function can be used to produce a confusion matrix 
    in order to determine how many observations were correctly or incorrectly 
    classified."""
    # print(confusion_matrix(y_test, pred).T)
    #ValueError: multilabel-indicator is not supported
    ##https://stackoverflow.com/questions/46953967/multilabel-indicator-is-not-supported-for-confusion-matrix
    # print(confusion_matrix(y_test.argmax(), pred.argmax(axis)).T)
    
    #or, written this better way:
    conf_m = pd.DataFrame(confusion_matrix(y_test, pred))
    conf_m.columns.name = 'Predicted'
    conf_m.index.name = 'True'
    conf_m
    print(conf_m)
    
    
    # https://medium.com/ai-in-plain-english/understanding-confusion-matrix-and-applying-it-on-knn-classifier-on-iris-dataset-b57f85d05cd8
    
    
    
    """The classification_report() function gives us some summary 
    statistics on the classifier's performance:"""

    # precision, recall, fbeta_score, support = precision_recall_fscore_support(y_test, knn_pred)
    # print(classification_report(y_test, logreg_pred, digits=3))
    
    # print("Sensitivity=Recall=Power: {}\nSpecificity=1-FPR: {}\nFPR: {}\nPrecision: {}".format())
    # print(Specificity)
    
    """ 
    the rows represent Actual Classes and the columns represent Class Predicted
    by the model. So the top left cell (483) represents the KNN model was successful
    at identifying 483 observations correctly labeled Blue Tarp, while 15 were 
    labeled Rooftop, 5 were labeled Soil, 1 were labeled Various Non-Tarp, and 0
    were labeled Vegetation
    while those mis-classification numbers are low, I need to ensure that they
    stay even lower because the consequences for mis-classification are severe
    (people in haiti earthquake hiding under blue tarps will not be found)
    """
#KNN
knn_confusion_matrix = conf_m(y_test,knn_pred)
knn_sensitivity = 0.9742063492063492 # 491/(491+13) TPR = Sensitivity = TP/(TP+FN) ... True Positive Rate (or Recall or Sensi
knn_specificity = 0.9983667603057425 #1-0.0016332396942575292 Specificity = 1 - FPR = TN/(TN+FP) = 15282/(15282+25)
knn_fpr = 0.0016332396942575292 # 25/(15282+25) = FPR = 1 - Specificity = FP/(TN+FP) ... False Positive Rate defines how many incorrect positive results occur among all negative samples available during the test.
knn_precision = 0.9515503875968992 # 491/(491+25) Precision = TruePositives / (TruePositives + FalsePositives)


#LDA
lda_confusion_matrix = conf_m(y_test,lda_pred)
lda_sensitivity = 0.8035714285714286 # 405/(405+99) TPR = Sensitivity = TP/(TP+FN) ... True Positive Rate (or Recall or Sensi
lda_specificity = 0.98915528843013 #1-0.010844711569869995 = Specificity = 1 - FPR = TN/(TN+FP) = 15141/(15141+166) 
lda_fpr = 0.010844711569869995 # 166/(15141+166) = FPR = 1 - Specificity = FP/(TN+FP) ... False Positive Rate defines how many incorrect positive results occur among all negative samples available during the test.
lda_precision = 0.7092819614711033 # 405/(405+166) Precision = TruePositives / (TruePositives + FalsePositives)


#QDA
qda_confusion_matrix = conf_m(y_test,qda_pred)
qda_sensitivity = 0.8511904761904762 # 429/(429+75) TPR = Sensitivity = TP/(TP+FN) ... True Positive Rate (or Recall or Sensi
qda_specificity = 0.9996080224733782 #1-0.00039197752662180704 = Specificity = 1 - FPR = TN/(TN+FP) = 15141/(15141+166)
qda_fpr = 0.00039197752662180704 # 6/(15301+6) = FPR = 1 - Specificity = FP/(TN+FP) ... False Positive Rate defines how many incorrect positive results occur among all negative samples available during the test.
qda_precision = 0.9862068965517241 # 429/(429+6) Precision = TruePositives / (TruePositives + FalsePositives)


#Logistic Regression
logistic_regression_confusion_matrix = conf_m(y_test,logreg_pred)
logreg_sensitivity = 0.8948412698412699 # 451/(451+53) TPR = Sensitivity = TP/(TP+FN) ... True Positive Rate (or Recall or Sensi
logreg_specificity = 0.9990853857712158 #1-0.0009146142287842164 Specificity = 1 - FPR = TN/(TN+FP) = 15293/(15293+14)
logreg_fpr = 0.0009146142287842164 # 14/(15293+14) = FPR = 1 - Specificity = FP/(TN+FP) ... False Positive Rate defines how many incorrect positive results occur among all negative samples available during the test.
logreg_precision = 0.9698924731182795 # 451/(451+14) Precision = TruePositives / (TruePositives + FalsePositives)


#Bagging
bagging_confusion_matrix = conf_m(y_test,bagging_pred)
bagging_sensitivity = 0.9471428571428572 # 663/(663+37) TPR = Sensitivity = TP/(TP+FN) ... True Positive Rate (or Recall or Sensi
bagging_specificity = 0.9985126425384234 #1-0.0009146142287842164 Specificity = 1 - FPR = TN/(TN+FP) = 20140/(20140+30)
bagging_fpr = 0.001487357461576599 # 30/(20140+30) = FPR = 1 - Specificity = FP/(TN+FP) ... False Positive Rate defines how many incorrect positive results occur among all negative samples available during the test.
bagging_precision = 0.9567099567099567 # 663/(663+30) Precision = TruePositives / (TruePositives + FalsePositives)


### alternatively
#test is 1-accuracy, so:
# error_test = 1 - svc.score(X_test, y_test)
# error_train = 1 - svc.score(X_train, y_train)








# Best Threshold=0.250000 #KNN
# Best Threshold=0.003224 # LDA
# Best Threshold=0.029901 #QDA
# Best Threshold=0.025541 #LogReg




### Filling it out

Bagging Accuracy: 0.9982605945604048
# Bagging Test Error: 0.0017394054395951652
auc_Bagging: 0.9819070785132629
Best Threshold=0.070379#bagging
bagging_confusion_matrix = conf_m(y_test,bagging_pred)
# Predicted     0    1
# True                
# 0          6123    4
# 1             7  190
bagging_sensitivity = 0.9471428571428572 # 663/(663+37) TPR = Sensitivity = TP/(TP+FN) ... True Positive Rate (or Recall or Sensi
bagging_specificity = 0.9985126425384234 #1-0.0009146142287842164 Specificity = 1 - FPR = TN/(TN+FP) = 20140/(20140+30)
bagging_fpr = 0.001487357461576599 # 30/(20140+30) = FPR = 1 - Specificity = FP/(TN+FP) ... False Positive Rate defines how many incorrect positive results occur among all negative samples available during the test.
bagging_precision = 0.9567099567099567 # 663/(663+30) Precision = TruePositives / (TruePositives + FalsePositives)




SVM Accuracy: 0.9876237623762376
# SVM Test Error: 0.012376237623762387
auc_SVM: 0.9871794871794871
Best Threshold=Best Threshold=0.024627 #SVM
svm_confusion_matrix = conf_m(y_test, clf_svm_pred)
# Predicted     0    1
# True                
# 0          6117   10
# 1            10  187
svm_sensitivity = 0.9471428571428572 # 663/(663+37) TPR = Sensitivity = TP/(TP+FN) ... True Positive Rate (or Recall or Sensi
svm_specificity = 0.9985126425384234 #1-0.0009146142287842164 Specificity = 1 - FPR = TN/(TN+FP) = 20140/(20140+30)
svm_fpr = 0.001487357461576599 # 30/(20140+30) = FPR = 1 - Specificity = FP/(TN+FP) ... False Positive Rate defines how many incorrect positive results occur among all negative samples available during the test.
svm_precision = 0.9567099567099567 # 663/(663+30) Precision = TruePositives / (TruePositives + FalsePositives)