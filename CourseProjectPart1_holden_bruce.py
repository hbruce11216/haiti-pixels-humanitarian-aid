#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 11 13:46:16 2020

@author: holdenbruce
"""

https://machinelearningmastery.com/threshold-moving-for-imbalanced-classification/
https://machinelearningmastery.com/roc-curves-and-precision-recall-curves-for-classification-in-python/
https://machinelearningmastery.com/tutorial-to-implement-k-nearest-neighbors-in-python-from-scratch/
https://datageneralist.files.wordpress.com/2018/03/master_machine_learning_algo_from_scratch.pdf
https://jupyter-notebook.readthedocs.io/en/stable/examples/Notebook/Working%20With%20Markdown%20Cells.html



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





## Load Data
pixels = pd.read_csv('HaitiPixels.csv', na_values=["?"])
def load_data(dataset):
    dataset.Class = dataset.Class.astype('category')
    
    #set X and Y and then create train and test data from the dataset
    X = pixels.iloc[:,1:4] #X is columns: Red, Green, Blue
    Y = pixels.Class  #Y is the Class column
    #this remapping turns Blue Tarp into classification=1 while every other choice 
    #is mapped to 0...thus making it a binary 
    Y = Y.map({'Blue Tarp':0, 'Rooftop':1,'Soil':1,'Various Non-Tarp':1, 'Vegetation':1})

    return X,Y
X,Y = load_data(pixels)
X.shape,Y.shape

sns.scatterplot(X.Red.values,X.Green.values,X.Blue.values)





## Test-Train Split
def train_test(X,Y):      
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=42) #split 75% for train and 25% test
    # X_train #A matrix containing the predictors associated with the training data (47,430 pixels)
    # X_test #A matrix containing the predictors associated with the test data (15,811 pixels)
    # y_train #A vector containing the class labels for the training observations, labeled Y_train below.
    return X_train, X_test, y_train, y_test
X_train, X_test, y_train, y_test = train_test(X,Y)










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











