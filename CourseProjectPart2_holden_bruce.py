#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 13 11:08:40 2020

@author: holdenbruce
"""


import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle

from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
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



## this sets the precision and general format for data output in this project
## one of the pieces of feedback i got from part 1 of this project was that 
## readability could be improved (primarily from me returning too many decimals
## so that it was making it hard to read tables), so i've chosen to specify the
## precision of decimals returned in pandas to 3
pd.set_option('precision', 3) # number precision for pandas
pd.set_option('display.max_rows', 12)
pd.set_option('display.max_columns', 12)
pd.set_option('display.float_format', '{:20,.4f}'.format) # get rid of scientific notation
plt.style.use('seaborn') # pretty matplotlib plots





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

#explore the shape of X and y
X.shape,y.shape #((63241, 3), (63241,))
pixels.head()

#rewrite the Class column in pixels with the new mapped version stored in y
pixels['Class'] = y
pixels.head() #looks good 
pixels.shape #(63241, 4)

#let's see what type of data is in each column
pixels.dtypes
# Class    category
# Red         int64
# Green       int64
# Blue        int64
# dtype: object

pixels.info() #get info on variables, looking for dtypes, could also use .dtypes

#check the variables of type object
X.Red.unique() #
X.Green.unique() #
X.Blue.unique() #
y.unique() #array([1, 0])



# https://medium.com/@hjhuney/implementing-a-random-forest-classification-model-in-python-583891c99652
# Random forests tend to shine in scenarios where a model has a large number of features 
# that individually have weak predicative power but much stronger power collectively¹.


# We’ll use train-test-split to split the data into training data and testing data.
## Test-Train Split
def train_test(X,y):      
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42) #split 75% for train and 25% test
    # X_train #A matrix containing the predictors associated with the training data (47,430 pixels)
    # X_test #A matrix containing the predictors associated with the test data (15,811 pixels)
    # y_train #A vector containing the class labels for the training observations, labeled Y_train below.
    return X_train, X_test, y_train, y_test
X_train, X_test, y_train, y_test = train_test(X,y)


# Now we can create the random forest model
## Bagging:
def bagging(X_train, X_test, y_train, y_test):
    # fit a model
    model = RandomForestClassifier(max_features=X_train.shape[1],random_state=313)
    
    #predict classification
    bagging_pred = model.fit(X_train, y_train).predict(X_test)
    
    # predict probabilities
    bagging_probs = model.fit(X_train, y_train).predict_proba(X_test)
    
    # keep probabilities for the positive outcome only
    bagging_probs = bagging_probs[:, 1]

    return bagging_pred, bagging_probs, model
bagging_pred, bagging_probs, model = bagging(X_train, X_test, y_train, y_test)


# Let’s next evaluate how the model performed.

# Evaluating Performance:
    
def eval_perform_rf():
    bagging_cv_score = cross_val_score(model, X, y, cv=10, scoring='roc_auc', n_jobs=-1)
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
    
eval_perform_rf()
    
# The confusion matrix is useful for giving you false positives and false negatives. 
# The classification report tells you the accuracy of your model. 
# The ROC curve plots out the true positive rate versus the false positive rate at various thresholds. 
# The roc_auc scoring used in the cross-validation model shows the area under the ROC curve.

# We’ll evaluate our model’s score based on the roc_auc score (stored in bagging_cv_score.mean()
#and returned as "Mean AUC Score - Rnadom Forest"), which is 0.9886

# The next thing we should do is tune our hyperparameters to see if we can improve 
# the performance of the model.
# https://towardsdatascience.com/hyperparameter-tuning-the-random-forest-in-python-using-scikit-learn-28d2aa77dd74

# Tuning Hyperparameters
# I’d recommend William Koehrsen’s article, “Hyperparameter Tuning the Random Forest in Python” for a more detailed description of the process. We’ll do a cliff-notes version.
# We’ll use RandomizedSearchCV from sklearn to optimize our hyperparamaters. 
# Koehrsen uses a full grid of hyperparameters in his article, but I found that this could take a very substantial time to run in practice. I decided to focus on 3 hyperparameters: n_estimators, max_features, and max_depth.

def rf_tune_hyperparameters():
    #n_estimators determines the number of trees in the random forest
    #take 11 values of n_estimators starting from 100 and ending with 2000, equally spaced 
    n_estimators = [int(x) for x in np.linspace(start = 100, stop = 2000, num = 11)] #default is 100
    # number of features at every split
    max_features = ['auto', 'sqrt','log2'] #default is auto in sklearn 
    
    # max_depth determines the maximum depth of the tree
    #take 11 values of max_depth starting from 100 and ending with 500, equally spaced 
    max_depth = [int(x) for x in np.linspace(100, 500, num = 11)]
    max_depth.append(None) #None is the default for max_depth in sklearn 
    # create random grid of these parameters 
    param_grid = {
     'n_estimators': n_estimators,
     'max_features': max_features,
     'max_depth': max_depth
     }
    
    # Random search of parameters
    rfc_random = RandomizedSearchCV(estimator = model, param_distributions = param_grid, n_iter = 100, cv = 3, verbose=2, random_state=313, n_jobs = -1)
    # Fit the model
    rfc_random.fit(X_train, y_train)
    #this is a pretty data intensive model fit which is why i have set n_jobs to -1
    #in the RandomizedSearchCV(), which means that all of my processors are being 
    #used in parallel to run this fit

    # print results
    # print(rfc_random.best_params_)
    
    #now store the optimal parameters and then return them for use in other functions
    optimal_n_estimators = rfc_random.best_params_['n_estimators']
    optimal_max_features = rfc_random.best_params_['max_features']
    optimal_max_depth = rfc_random.best_params_['max_depth']

    return rfc_random, optimal_n_estimators, optimal_max_features, optimal_max_depth

rfc_random, optimal_n_estimators, optimal_max_features, optimal_max_depth = rf_tune_hyperparameters()


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

rf_opimal_cv_score = rf_optimal()







### SVM

len(pixels) #63241
pixels.head()

## Downsample the data 
# 63241 samples will take a long time for the support vector machine to run
# so downsampling will make this process quicker...
# Support Vector Machines are great with small datasets, but not awesome with large ones, 
# and this dataset, while not huge, is big enough to take a long time to optimize with 
# Cross Validation. So we'll downsample both categories, Blue Tarp and Non Blue Tarp
# to 1,000 each.
#### - from statquest support vector machine

def downsample_data(num_samples):    
    #to ensure that i actually get 1000 samples for each ategory, i'll start by 
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
X_train_downsampled, X_test_downsampled, y_train_downsampled, y_test_downsampled = train_test(X_downsampled, y_downsampled)

#then we will scale the data using StandardScaler 
scaler = preprocessing.StandardScaler().fit(X_train_downsampled)
X_train_downsampled_scaled = scaler.transform(X_train_downsampled)
X_test_downsampled_scaled = scaler.transform(X_test_downsampled)



def build_svm(X_train_foo_scaled, X_test_foo_scaled, y_train_foo):

    #now that we have our data split into test and train we can build our preliminary 
    #support vector machine
    clf_svm = SVC(random_state=313)
    clf_svm.fit(X_train_downsampled_scaled, y_train_downsampled)

    #that's it, that's our support vector machine for classification
    #now we can draw a confusion matrix and see how it performs on the test data 

    plot_confusion_matrix(clf_svm, 
                      X_test_downsampled_scaled, 
                      y_test_downsampled,
                      values_format='d',
                      display_labels=["Not Blue Tarp", "Blue Tarp"])
#this performs incredibly well 
#the confusmion matrix shows us that of the 264 observations that were 
#not blue tarp, 258 (97.7%) were correctly classified
#of the 236 observatiosn that were blue tarps, 235 (99.6%) were correctly
#classified 
#this goes to show how great SVM is as an out-of-the-box solution / machine
#learning technique...
#i will use cross validation to optimize the parameters to see if i can improve
#predictions at all but honestly, these numbers are already fantastic...my next
#step would be to run the entire dataset (without downsampling) to see how it
#performs 


def optimizing_svm():
    #when we are optimizing a support vector machine we are attempting to find the 
    #best value for gamma and the regularization parameter C that will improve the
    #prediction/classification accuracy of the model 
    #since we are looking to optimize two parameters (gamma and C) we will use 
    #GridSearchCV(), specifying several values for gamma and C and then letting
    #the function determine the optimal combination (that's the great thing about using
    #GridSerachCV() for problems like this: it tests all possible combinations of 
    #parameters for us and all we have to do is plug in some options)
    
    #the default values for the SVC parameters in sklearn.svm are:
    #C=1.0
    #kernel='rbf
    #degree=3
    #gamma='scale' (which is 1/(n_features*X.var()) while 'auto' uses 1/n_features
    
    #GridSearchSV in sklearn.model_selection has a parameter called param_grid
    #where you can pass in a dictionary of parameters 
    #i will pass in all of the default SVC parameters and then some extras
    #so that the GridSearchCV() can find the optimal combination from many options
    
    #i might not need to have all these options for kernel because my assumption
    #is that 'rbf' will work the best for the pixel data, but i am including 
    #'linear','poly', and 'sigmoid' 
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


#now fit the model with the optimal parameter grid ... searching through best options
optimal_params.fit(X_train_downsampled_scaled, y_train_downsampled)
#takes a little while 

#now print the optimal parameters for the model
print('''The best cost for clf_svm is: {}
      the best degree for clf_svm is: {}
      the best gamma for clf_svm is: {}
      the best kernel for clf_svm is: '{}'
      '''.format(optimal_params.best_params_['C'],
      optimal_params.best_params_['degree'],
      optimal_params.best_params_['gamma'],
      optimal_params.best_params_['kernel']
)) # best C=0.901, degree=1, gamma=1, kernel='rbf'

    
#then use those optimal parameters to fit a new model 
clf_svm_optimal = SVC(random_state=313, C=0.901, degree=1, gamma=1, kernel='rbf')
clf_svm_optimal.fit(X_train_downsampled_scaled, y_train_downsampled)

#now i'll draw a confusion matrix for this optimal svc model to see if 
#tuning the parameters improved the support vector machine any more than
#the original fit, which was already very good
plot_confusion_matrix(clf_svm_optimal, 
                      X_test_downsampled_scaled, 
                      y_test_downsampled,
                      values_format='d',
                      display_labels=["Not Blue Tarp", "Blue Tarp"])
#this performs identically (incredibly) well 
#the confusmion matrix shows us that of the 264 observations that were 
#not blue tarp, 258 (97.7%) were correctly classified
#of the 236 observatiosn that were blue tarps, 235 (99.6%) were correctly
#classified 
#this goes to show how great SVM is as an out-of-the-box solution / machine
#learning technique...despite using cross validation to optimize the parameters 
#to improve predictions, these numbers were already optimal from the start











#the next step would still be to run the entire dataset (without downsampling) 
#to see how it performs 

pixels.info()
pixels.head()
X,y

# We’ll use the train_test() function we wrote earlier to split the data into 
# training data and testing data
X_train_full, X_test_full, y_train_full, y_test_full = train_test(X, y)

#then we will scale the data using StandardScaler 
scaler = preprocessing.StandardScaler().fit(X_train_full)
X_train_full_scaled = scaler.transform(X_train_full)
X_test_full_scaled = scaler.transform(X_test_full)

#now that we have our data split into test and train we can build our preliminary 
#support vector machine
clf_svm = SVC(random_state=313)
clf_svm.fit(X_train_full_scaled, y_train_full)

#that's it, that's our support vector machine for classification
#now we can draw a confusion matrix and see how it performs on the test data 

plot_confusion_matrix(clf_svm, 
                      X_test_full_scaled, 
                      y_test_full,
                      values_format='d',
                      display_labels=["Not Blue Tarp", "Blue Tarp"])












#split this data using the load_data() from above and pass it the downsampled data
X_downsampled, y_downsampled = load_data(pixels_downsampled)
#nothing to one hot encode in X





#set X and Y and then create train and test data from the dataset
X_full = pixels.iloc[:,1:4] #X is columns: Red, Green, Blue
y_full = pixels.Class  #Y is the Class column





#these variables contain data for the entire pixels dataset instead of just 
#the 1000 downsampled version i used above to test and create the optimal SVM
X_full.shape 
y_full.shape









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








# Running the example fits a logistic regression model on the training dataset then evaluates it using a range of thresholds on the test set, creating the ROC Curve
# We can see that there are a number of points or thresholds close to the top-left of the plot.
# Which is the threshold that is optimal?

# There are many ways we could locate the threshold with the optimal balance between false positive and true positive rates.
# Firstly, the true positive rate is called the Sensitivity. The inverse of the false-positive rate is called the Specificity.
# Sensitivity = TruePositive / (TruePositive + FalseNegative)
# Specificity = TrueNegative / (FalsePositive + TrueNegative)
# Where:
# Sensitivity = True Positive Rate
# Specificity = 1 – False Positive Rate
# The Geometric Mean or G-Mean is a metric for imbalanced classification that, if optimized, will seek a balance between the sensitivity and the specificity.

# G-Mean = sqrt(Sensitivity * Specificity)
# One approach would be to test the model with each threshold returned from the call roc_auc_score() and select the threshold with the largest G-Mean value.

# Given that we have already calculated the Sensitivity (TPR) and the complement to the Specificity when we calculated the ROC Curve, we can calculate the G-Mean for each threshold directly.

# The threshold is then used to locate the true and false positive rates, then this point is drawn on the ROC Curve.

# It turns out there is a much faster way to get the same result, called the Youden’s J statistic.
# The statistic is calculated as:
# J = Sensitivity + Specificity – 1
# Given that we have Sensitivity (TPR) and the complement of the specificity (FPR), we can calculate it as:
# J = Sensitivity + (1 – FalsePositiveRate) – 1
# Which we can restate as:
# J = TruePositiveRate – FalsePositiveRate
# We can then choose the threshold with the largest J statistic value. For example:
    
#### BUT THEN WHAT???
### now that we have identified the optimal threshold...what do i do with it?
### this is clearly the limit of my understanding, what good is it to identify the 
## threshold if we don't then use it to actually tune the LogisticRegression. We are
## identifying the optimal threshold after running the LogisticRegression and then
## don't apply it anywhere 















# https://towardsdatascience.com/random-forest-in-python-24d0893d51c0

pixels.shape
pixels.columns
pixels.head()

# now we'll split the data into X (predictors) and y (response)
# we want to predict whether the customer purchased a caravan insurance 
# policy (Purchase), using the other 85 columns as predictors 

# i'm using copy() here so that if i make a mistake when making my classification
# trees, i won't accidentally overwrite the original dataset and can just 
# reload X instead of reloading the entire caravan csv
X = pixels.drop('Class',axis=1).copy()
X.head()
X.info()
# as discussed above, all of the X columns are type int64 so we don't need
# to worry about one-hot encoding any of the values to makes ure XGBoost runs

y = pixels['Class'].copy()
y.head()
y.unique()
#y, on the other hand, is type object with 5 values
#but really it's only 2 values: 'Blue Tarp' and 'Not Blue Tarp',
#so we need to map 'Blue Tarp' to 1 and 'Not Blue Tarp' to 0
y = y.map({'Vegetation':0, 'Soil':0, 'Rooftop':0, 'Various Non-Tarp':0, 'Blue Tarp':1})
y.head()
y.unique()
#now we're good 

# https://medium.com/@hjhuney/implementing-a-random-forest-classification-model-in-python-583891c99652
# Random forests tend to shine in scenarios where a model has a large number of features 
# that individually have weak predicative power but much stronger power collectively¹.

# We’ll use train-test-split to split the data into training data and testing data.
from sklearn.model_selection import train_test_split
# implementing train-test-split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=66)

# Now, we can create the random forest model.
from sklearn import model_selection

## Bagging
from sklearn.ensemble import RandomForestClassifier

def bagging(X_train, X_test, y_train, y_test):
    # fit a model
    model = RandomForestClassifier(max_features=X_train.shape[1],random_state=313)
    
    #predict classification
    bagging_pred = model.fit(X_train, y_train).predict(X_test)
    
    # predict probabilities
    bagging_probs = model.fit(X_train, y_train).predict_proba(X_test)
    
    # keep probabilities for the positive outcome only
    bagging_probs = bagging_probs[:, 1]

    return bagging_pred, bagging_probs, model
bagging_pred, bagging_probs, model = bagging(X_train, X_test, y_train, y_test)





# random forest model creation
# rfc = RandomForestClassifier()
# rfc.fit(X_train,y_train)
# # predictions
# rfc_predict = rfc.predict(X_test)






# Let’s next evaluate how the model performed.

# Evaluating Performance
# We’ll import cross_val_score, classification_report, and confusion_matrix.
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report, confusion_matrix

# We’ll also run cross-validation to get a better overview of the results.
# rfc_cv_score = cross_val_score(rfc, X, y, cv=10, scoring='roc_auc')
bagging_cv_score = cross_val_score(model, X, y, cv=10, scoring='roc_auc')

# Now, we’ll print out the results.
# print(rfc_cv_score)
# for i in range(len(rfc_cv_score)):
#     print('CV={}: {}'.format(i+1,rfc_cv_score[i]))
print(bagging_cv_score)
for i in range(len(bagging_cv_score)):
    print('CV={}: {}'.format(i+1,bagging_cv_score[i]))

print("=== Confusion Matrix ===")
# print(confusion_matrix(y_test, rfc_predict))
print(confusion_matrix(y_test, bagging_pred))
print('\n')







print("=== Classification Report ===")
# print(classification_report(y_test, rfc_predict))
print(classification_report(y_test, bagging_pred))
print('\n')
print("=== All AUC Scores ===")
print(rfc_cv_score)
print('\n')
print("=== Mean AUC Score ===")
print("Mean AUC Score - Random Forest: ", rfc_cv_score.mean())

# === Confusion Matrix ===
# [[20143    27]
#  [   41   659]]


# === Classification Report ===
#               precision    recall  f1-score   support

#            0       1.00      1.00      1.00     20170
#            1       0.96      0.94      0.95       700

#     accuracy                           1.00     20870
#    macro avg       0.98      0.97      0.97     20870
# weighted avg       1.00      1.00      1.00     20870



# === All AUC Scores ===
# [1.         0.98507331 0.9998969  1.         1.         1.
#  0.99885456 0.99900214 0.94035147 0.99945073]


# === Mean AUC Score ===
# Mean AUC Score - Random Forest:  0.992262910839081




# The confusion matrix is useful for giving you false positives and false negatives. 
# The classification report tells you the accuracy of your model. 
# The ROC curve plots out the true positive rate versus the false positive rate at various thresholds. 
# The roc_auc scoring used in the cross-validation model shows the area under the ROC curve.

# We’ll evaluate our model’s score based on the roc_auc score, which is .792. 
# The next thing we should do is tune our hyperparameters to see if we can improve 
# the performance of the model.
# https://towardsdatascience.com/hyperparameter-tuning-the-random-forest-in-python-using-scikit-learn-28d2aa77dd74

# Tuning Hyperparameters
# I’d recommend William Koehrsen’s article, “Hyperparameter Tuning the Random Forest in Python” for a more detailed description of the process. We’ll do a cliff-notes version.
# We’ll use RandomizedSearchCV from sklearn to optimize our hyperparamaters. 
# Koehrsen uses a full grid of hyperparameters in his article, but I found that this could take a very substantial time to run in practice. I decided to focus on 3 hyperparameters: n_estimators, max_features, and max_depth.

from sklearn.model_selection import RandomizedSearchCV
# number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 100, stop = 2000, num = 10)] #default is 100
# number of features at every split
max_features = ['auto', 'sqrt','log2'] #default is auto in sklearn 

# max depth
max_depth = [int(x) for x in np.linspace(100, 500, num = 11)]
max_depth.append(None) #None is the default for max_depth in sklearn 
# create random grid
random_grid = {
 'n_estimators': n_estimators,
 'max_features': max_features,
 'max_depth': max_depth
 }
# Random search of parameters
rfc_random = RandomizedSearchCV(estimator = model, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)
# Fit the model
rfc_random.fit(X_train, y_train)
#this is a pretty data intensive model fit which is why i have set n_jobs to -1
#in the RandomizedSearchCV(), which means that all of my processors are being 
#used in parallel to run this fit



#Output:
# Fitting 3 folds for each of 100 candidates, totalling 300 fits
# [Parallel(n_jobs=-1)]: Using backend LokyBackend with 12 concurrent workers.
# [Parallel(n_jobs=-1)]: Done  17 tasks      | elapsed:   30.9s
# [Parallel(n_jobs=-1)]: Done 138 tasks      | elapsed:  3.8min
# [Parallel(n_jobs=-1)]: Done 300 out of 300 | elapsed:  7.9min finished
# Out[114]: 
# RandomizedSearchCV(cv=3, estimator=RandomForestClassifier(), n_iter=100,
#                    n_jobs=-1,
#                    param_distributions={'max_depth': [100, 140, 180, 220, 260,
#                                                       300, 340, 380, 420, 460,
#                                                       500, None],
#                                         'max_features': ['auto', 'sqrt'],
#                                         'n_estimators': [200, 400, 600, 800,
#                                                          1000, 1200, 1400, 1600,
#                                                          1800, 2000]},
#                    random_state=42, verbose=2)

#this takes quite a while to run 

# print results
print(rfc_random.best_params_)
#first attempt:
# {'n_estimators': 200, 'max_features': 'sqrt', 'max_depth': 140}

# The best results were n_estimators: 200, max_features: sqrt, max_depth: 140
# now we can plug these back into the model to see if it improved our performance

rfc = RandomForestClassifier(n_estimators=200, max_depth=140, max_features='sqrt')
rfc.fit(X_train,y_train)
rfc_predict = rfc.predict(X_test)
rfc_cv_score = cross_val_score(rfc, X, y, cv=10, scoring='roc_auc')
print("=== Confusion Matrix ===")
print(confusion_matrix(y_test, rfc_predict))
print('\n')
print("=== Classification Report ===")
print(classification_report(y_test, rfc_predict))
print('\n')
print("=== All AUC Scores ===")
print(rfc_cv_score)
print('\n')
print("=== Mean AUC Score ===")
print("Mean AUC Score - Random Forest: ", rfc_cv_score.mean())

# === Confusion Matrix ===
# [[20145    25]
#  [   43   657]]


# === Classification Report ===
#               precision    recall  f1-score   support

#            0       1.00      1.00      1.00     20170
#            1       0.96      0.94      0.95       700

#     accuracy                           1.00     20870
#    macro avg       0.98      0.97      0.97     20870
# weighted avg       1.00      1.00      1.00     20870



# === All AUC Scores ===
# [1.         0.98833375 0.99992682 1.         1.         1.
#  0.9989334  0.99652163 0.93982019 0.99922821]


# === Mean AUC Score ===
# Mean AUC Score - Random Forest:  0.9922764003412443



## this is an almost negligible difference from the last one 






#second attempt:
#{'n_estimators': 1155, 'max_features': 'sqrt', 'max_depth': 460}

# The best results were n_estimators: 1155, max_features: sqrt, max_depth: 460
# now we can plug these back into the model to see if it improved our performance

rfc_optimal = RandomForestClassifier(n_estimators=1155, max_depth=460, max_features='sqrt', n_jobs=-1)
rfc_optimal.fit(X_train,y_train)
rfc_optimal_predict = rfc_optimal.predict(X_test)
rfc_optimal_cv_score = cross_val_score(rfc_optimal, X, y, cv=10, scoring='roc_auc', n_jobs=-1)

print("=== Confusion Matrix ===")
print(confusion_matrix(y_test, rfc_predict))
print('\n')
# === Confusion Matrix ===
# [[20145    25]
#  [   41   659]]

print("=== Classification Report ===")
print(classification_report(y_test, rfc_predict))
print('\n')
print("=== All AUC Scores ===")
print(rfc_cv_score)
print('\n')
print("=== Mean AUC Score ===")
print("Mean AUC Score - Random Forest: ", rfc_cv_score.mean())
