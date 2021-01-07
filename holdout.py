#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 17 15:22:24 2020

@author: holdenbruce
"""


# now we are going to use the holdout dataset to train the model
# and then use the haitipixels.csv as the test dataset

# so the train will be huge and will be 100% from the holdout dataset
# and the test set will be 100% from the haitipixels dataset

import pandas as pd

## Load Data
pixels = pd.read_csv('HaitiPixels.csv', na_values=["?"])
holdout = pd.read_csv('concat_data.csv', na_values=["?"])
holdout.head()
holdout.info()


pixels.head() #print first 5 

def load_data(dataset):
    
    #set X and y and then create train and test data from the dataset
    X = dataset.drop(['Class'], axis=1) #X is columns: Red, Green, Blue

    y = dataset.Class #Y is the Class column
    # if 'Class' in dataset.columns:
    #     y = dataset.Class #Y is the Class column
    # if 'tarp' in dataset.columns:
    #     y = dataset.tarp 
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

# the pixels dataset will be used for testing
X_pixels, y_pixels = load_data(pixels)

#the holdout dataset will be used for training
X_holdout, y_holdout = load_data(holdout)


# there is no use for calling kfold_train_test_split() because the train is
# the holdout data and the test is the pixels data
X_train, y_train = X_holdout, y_holdout
X_test, y_test = X_pixels, y_pixels



# however we still need to scale our data
# this will help decrease run time on the SVM
# and is important for ... 
def scale_X_data(X_train_foo, X_test_foo):
    #then we will scale the data using StandardScaler 
    scaler = preprocessing.StandardScaler().fit(X_train_foo)
    X_train_foo_scaled = scaler.transform(X_train_foo)
    X_test_foo_scaled = scaler.transform(X_test_foo)
    
    return X_train_foo_scaled, X_test_foo_scaled

X_train_scaled, X_test_scaled = scale_X_data(X_train, X_test)






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

import time # Just to compare fit times
start = time.time()
# using scaled X data for the bagging 
bagging_pred, bagging_probs, rfc = bagging(X_train_scaled, X_test_scaled, y_train, y_test)
end = time.time()
print("Tune Fit Time:", end - start)
# Tune Fit Time: 464.0315408706665





def eval_perform_rf(rfc, X, y, y_test, bagging_pred):
    bagging_cv_score = cross_val_score(rfc, X, y, cv=10, scoring='roc_auc', n_jobs=-1)
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
    
    
#first time around:
    
# we pass X_pixels and y_pixels into this eval_perform_rf() because we are 
# using this to test the performance of the random forest on the test data
eval_perform_rf(rfc, X_pixels, y_pixels, y_test, bagging_pred)

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
# [[61215     4]
#  [  613  1409]]


# === Classification Report ===
#               precision    recall  f1-score   support

#            0       0.99      1.00      0.99     61219
#            1       1.00      0.70      0.82      2022

#     accuracy                           0.99     63241
#    macro avg       0.99      0.85      0.91     63241
# weighted avg       0.99      0.99      0.99     63241



# === All AUC Scores ===
# [1.         0.99994825 0.99752475 1.         1.         1.
#  0.98687213 0.99401323 0.90815344 0.99935416]


# === Mean AUC Score ===
# Mean AUC Score - Random Forest:  0.9885865958789907


# We predicted that not blue tarp for 613 observations where it really was 
# a blue tarp
# and only predicted 4 blue tarps that really were not blue tarps
# This honestly is not ideal. I would like this to be the opposite.
# The idea of people not being found outweighs thinking blue tarps are there
# when they really aren't. 




#second time around

eval_perform_rf(rfc, X_train_scaled, y_train, y_test, bagging_pred)

# CV=1: 1.0
# CV=2: 1.0
# CV=3: 1.0
# CV=4: 1.0
# CV=5: 0.9999999920308404
# CV=6: 0.9976446365271635
# CV=7: 0.9900376606542723
# CV=8: 0.9741268696549095
# CV=9: 0.9944207304130871
# CV=10: 0.9866697081245687


# === Confusion Matrix ===
# [[61214     5]
#  [  611  1411]]


# === Classification Report ===
#               precision    recall  f1-score   support

#            0       0.99      1.00      0.99     61219
#            1       1.00      0.70      0.82      2022

#     accuracy                           0.99     63241
#    macro avg       0.99      0.85      0.91     63241
# weighted avg       0.99      0.99      0.99     63241



# === All AUC Scores ===
# [1.         1.         1.         1.         0.99999999 0.99764464
#  0.99003766 0.97412687 0.99442073 0.98666971]


# === Mean AUC Score ===
# Mean AUC Score - Random Forest:  0.9942899597404841









## random forest hyperparameters tuning 
    
# https://www.analyticsvidhya.com/blog/2020/03/beginners-guide-random-forest-hyperparameter-tuning/
# this guide was instrumental in developing my understanding of which hyperparameters 
# really matter when tuning a random forest model
# initially, my approach was to throw in a bunch of parameters into the param_grid
# and just let the model identify the optimal parameters through the use of GridSearchCV
# but this quickly proved to be unscalable 
# i tried upgrading this with RandomSearchCV but the big 2mil dataset pummeled 
# any hopes I had of using my original tuning function to fit the optimal random forest
# model... i needed to find a better solution to deal with large datasets 
# in this guide from the analyticsvidhya.com article, i identified a few key
# parameters to focus on and learned how a much smaller range of options was 
# necessary to pass to the gridsearch than I had initially thought

# here is a synopsis of my rationale for choosing the parameters that i did:

# max_depth determines the limit of the depth for each tree in the random forest
# the performance of the model on training data increases continuously as max_depth
# increases, because it gets closer and closer to a perfect fit of the data...
# simultaneously, the fit on test data will decrease as max_depth increases since
# the model is overfit to the training data and thus will perform poorly on test data

# min_sample_split determines the minimum number of observations for any node to split,
# by default this number is set to 2, which means that if any terminal node has more 
# than two observations and is not a pure node, it can be split further into subnodes...
# as discussed above for max_depth, we want to avoid a random forest model comprised of
# trees with too many nodes as that would indicate overfitting of the model.
# leaving this min_sample_split number set to its default of 2 allows for this overfitting
# to occur, since 2 is so small and allows for trees to continue splitting until all the 
# nodes are pure (1).
# "by increasing this number we can reduce the number of splits that happen in the decision
# tree and can thus prevent the model from overfitting" (or at least mitigate)
# however, it is important to not underfit this model (this is done by having the 
# min_sample_split be too high, which would essentially lead to there being no 
# significant splits observed, ultimately leading to a dip in both the training and
# test scores of model performance)

# max_terminal_nodes / max_leaf_nodes sets a condition on the splitting of the nodes
# in the tree, restricting the tree's growth. when the max_leaf_nodes is small, the 
# random forest modele will underfit.

# min_samples_leaf specifies the minimum number of samples that should be present
# in the leaf node after splitting a node ... not sure how helpful this will be tho

# n_estimators is super helpful for solving the exact problem i was having: as the
# number of trees used in the random forest model increases, so does the time complexity
# of the model...so by limiting the n_estimators, we can control the time complexity 
# from getting out of hand when working with large datasets

# max_samples determines what fraction of the original dataset is given to any 
# individual tree

# max_features determines the maximum number of features provided to each tree
# in the random forest model...the default value for this is set to the square
# root of the number of features present in the dataset .. and the ideal number
# of max_features generally tend to lie close to this value, so i'll be leaving
# the max_features set to its default and will not be using it in the gridsearch


# ex: 
param_rf = {
    'clf__n_estimators': [50, 100, 300],
    'clf__max_depth': [5, 8, 15, 25],
    'clf__min_samples_leaf': [1, 2, 5]
    }


def rf_tune_hyperparameters(rfc, X_train_rf_tune, y_train_rf_tune):
    #n_estimators determines the number of trees in the random forest
    n_estimators = [10, 50, 100]
    
    # max_depth determines the maximum depth of the tree
    max_depth = [4,8,12]

    #min_samples_split determines the number of observations needed to split a node
    min_samples_split = [5]
    
    # create grid of these parameters that will be passed into the searchCV function
    param_grid = {
     'n_estimators': n_estimators,
     'max_depth': max_depth,
     'min_samples_split': min_samples_split
     # 'max_features': max_features,
     }
    
    
    # GridSearchCV of the parameters
    rfc_gridsearch = GridSearchCV(
        rfc,
        param_grid,
        # verbose=2,
        n_jobs=-1
        )
    
    # # Random search of parameters
    # rfc_random = RandomizedSearchCV(
    #     estimator = model, 
    #     param_distributions = param_grid, 
    #     n_iter = 100, 
    #     cv = 3, 
    #     verbose=2, 
    #     random_state=313, n_jobs = -1)



    # Fit the model
    rfc_gridsearch.fit(X_train_rf_tune, y_train_rf_tune)
    
       
    # rfc_random.fit(X_train.to_numpy(), y_train.to_numpy())
    # this is a pretty data intensive model fit which is why i have set n_jobs to -1
    #in the RandomizedSearchCV(), which means that all of my processors are being 
    #used in parallel to run this fit

    # print results
    # print(rfc_random.best_params_)
    
    
     
    #now store the optimal parameters and then return them for use in other functions
    optimal_n_estimators = rfc_gridsearch.best_params_['n_estimators']
    optimal_max_depth = rfc_gridsearch.best_params_['max_depth']
    optimal_min_samples_split = rfc_gridsearch.best_params_['min_samples_split']
    # optimal_max_features = rfc_gridsearch.best_params_['max_features']
    

    
    print("=== Optimal n_estimators ===")
    print(optimal_n_estimators)
    print('\n')
    
    print("=== Optimal optimal_max_depth ===")
    print(optimal_max_depth)
    print('\n')
    
    print("=== Optimal min_samples_split ===")
    print(optimal_min_samples_split)
    print('\n')
    
    # print("=== Optimal max_depth ===")
    # print(optimal_max_depth)
    # print('\n')
    
    # return rfc_random, optimal_n_estimators, optimal_max_features, optimal_max_depth
    return rfc_gridsearch, optimal_n_estimators, optimal_max_depth, optimal_min_samples_split


#second time around:

rfc_random, optimal_n_estimators, optimal_max_depth, optimal_min_samples_split = rf_tune_hyperparameters(rfc, X_train_scaled, y_train)

# === Optimal n_estimators ===
# 10

# === Optimal optimal_max_depth ===
# 12

# === Optimal min_samples_split ===
# 5












#####
# TALK ABOUT MODEL DRIFT
# the error that comes along with using a model fit on one dataset to
# then use new data on that same model...because that new data might 
# not fit the same distributional set 

# bayesian ideology would partially solve for this
# but the problem is that using bayesian modeling on a downsampled dataset
# and then applying it to a larger dataset literally goes directly against 
# the idea of bayesian statistics: that your assumptions should change
# when new data is in play...




# do the KFold before scaling




# def my_failed_attempt():            
            
            
            
            # rfc_random, optimal_n_estimators, optimal_max_features, optimal_max_depth = rf_tune_hyperparameters(rfc, X_train, y_train)
            
            # # Fitting 10 folds for each of 10 candidates, totalling 100 fits
            # # [Parallel(n_jobs=-1)]: Using backend LokyBackend with 12 concurrent workers.
            # # /Users/holdenbruce/opt/anaconda3/lib/python3.7/site-packages/joblib/externals/loky/process_executor.py:691: UserWarning: A worker stopped while some jobs were given to the executor. This can be caused by a too short worker timeout or by a memory leak.
            # #   "timeout or by a memory leak.", UserWarning
            
            # # https://github.com/scikit-learn/scikit-learn/issues/12546
            # # one of the maintainers for scikit-learn posited that numpy arrays wouldn't
            # # cause this to error while pandas would
            # rfc_random, optimal_n_estimators, optimal_max_features, optimal_max_depth = rf_tune_hyperparameters(rfc, X_train.to_numpy(), y_train.to_numpy())
            
            # # several people also said that any version after 20.0 was throwing errors
            # # using GridSearchCV but version before that (in 19.x) didn't have those same 
            # # issues...so if the .to_numpy() doesn't fix this then i'm going to try downgrading
            # # my scikit-learn  version to see if that solves this timeout issue 
            
            # # https://joblib.readthedocs.io/en/latest/parallel.html
            # # actually, after reading the docs for joblib it seems like invoking threading
            # # instead of relying on python to control how many workers/jobs to use 
            # # might actually be more efficient 
            # # 
            # from joblib import Parallel, delayed
            # Parallel(n_jobs=2, prefer="threads")(
            #     delayed(rfc_random)
            #     )
            
            # # ^ i added this to the rf_tune_hyperparameters()
            # # lets call it again and see if that improves things
            
            
            # # rfc_random, optimal_n_estimators, optimal_max_features, optimal_max_depth = rf_tune_hyperparameters(rfc, X_train.to_numpy(), y_train.to_numpy())
            
            
            
            
            # # now we can plug these back into the model to see if it improved our performance
            # # def rf_optimal():
            # # rf_opimal_cv_score = rf_optimal()
            
            
            
            #     # https://towardsdatascience.com/machine-learning-gridsearchcv-randomizedsearchcv-d36b89231b10
            #     # In the Logistic Regression and the Support Vector Classifier, the parameter that determines the strength of the regularization is called C.
            #     # For a high C, we will have a less regularization and that means we are trying to fit the training set as best as possible. Instead, with low values of the parameter C, the algorithm tries to adjust to the “majority” of data points and increase the generalization of the model.
            #     # There is another important parameter called gamma. But before to talk about it, I think it is important to understand a little bit the limitation of linear models.
            #     # Linear models can be quite limiting in low-dimensional spaces, as lines and hyperplanes have limited flexibility. One way to make a linear model more flexible is by adding more features, for example, by adding interactions or polynomials of the input features.
            #     # A linear model for classification is only able to separate points using a line, and that is not always the better choice. So, the solution could be to represent the points in a three-dimensional space and not in a two-dimensional space. In fact, in three-dimensional space, we can create a plane that divides and classifies the points of our dataset in a more precise way.
            #     # There are two ways to map your data into a higher-dimensional space: the polynomial kernel, which computes all possible polynomials up to a certain degree of the original features; and the radial basis function(RBF) kernel, also known as the Gaussian kernel which measures the distance between data points. Here, the task of gamma is to control the width of the Gaussian Kernel.
            #     # ...
            #     # So, Grid Search is good when we work with a small number of hyperparameters. However, if the number of parameters to consider is particularly high and the magnitudes of influence are imbalanced, the better choice is to use the Random Search.
                
            #     # if i end up using GridSearchCV instead of RnadomizedSearchCV, check the random_forests.py 
                
            #     # ... from the same url above:
            #     # RandomizedSearchCV is very useful when we have many parameters to try and the training time is very long
                
                
                
                
                
                
                
            #     # Random search of parameters
            #     rfc_random = RandomizedSearchCV(estimator = rfc, param_distributions = param_grid, n_iter = 10, cv = 10, verbose=2, random_state=313, n_jobs = -2)
            #     # switching n_jobs to -2 so that it doesn't use up my entire CPU
            
            
            #     # This is the output...still taking insanely long to run so clearly i need
            #     # to find another solution to this problem...:
                
            #     # Fitting 10 folds for each of 10 candidates, totalling 100 fits
            #     # [CV] n_estimators=1000, max_features=1, max_depth=6 ..................[CV] n_estimators=1000, max_features=1, max_depth=6 ..................
                
            #     # [Parallel(n_jobs=2)]: Using backend ThreadingBackend with 2 concurrent workers.
            #     # [CV]  n_estimators=1000, max_features=log2, max_depth=4, total=586.0min
            #     # [CV] .. n_estimators=1000, max_features=1, max_depth=6, total=879.2min
            #     # [CV] .. n_estimators=1000, max_features=1, max_depth=6, total=879.4min
            #     # [CV] .. n_estimators=1000, max_features=1, max_depth=6, total=879.6min
            #     # [CV] .. n_estimators=1000, max_features=1, max_depth=6, total=879.6min
            #     # [CV] .. n_estimators=1000, max_features=1, max_depth=6, total=879.6min
            #     # [CV] .. n_estimators=1000, max_features=1, max_depth=6, total=879.7min
            #     # [CV] .. n_estimators=1000, max_features=1, max_depth=6, total=879.7min
            #     # [CV] .. n_estimators=1000, max_features=1, max_depth=6, total=879.8min
            #     # [CV] .. n_estimators=1000, max_features=1, max_depth=6, total=879.8min
            #     # [CV] .. n_estimators=1000, max_features=1, max_depth=6, total=879.8min
            #     # [CV] .. n_estimators=1000, max_features=1, max_depth=6, total=879.2min
            #     # [CV] n_estimators=1000, max_features=1, max_depth=6 ..................
            #     # [CV] .. n_estimators=1000, max_features=1, max_depth=6, total=879.2min
            #     # [CV] n_estimators=1000, max_features=1, max_depth=6 ..................
            #     # [CV] ... n_estimators=1000, max_features=1, max_depth=6, total=18.4min
            #     # [CV] n_estimators=1000, max_features=1, max_depth=6 ..................
            #     # [CV] ... n_estimators=1000, max_features=1, max_depth=6, total=18.4min
            #     # [CV] n_estimators=1000, max_features=1, max_depth=6 ..................
            #     # [CV] ... n_estimators=1000, max_features=1, max_depth=6, total=38.7min
            #     # [CV] n_estimators=1000, max_features=1, max_depth=6 ..................
            #     # [CV] ... n_estimators=1000, max_features=1, max_depth=6, total=38.8min
            #     # [CV] n_estimators=1000, max_features=1, max_depth=6 ..................
            #     # [CV] ... n_estimators=1000, max_features=1, max_depth=6, total=53.3min
            #     # [CV] n_estimators=1000, max_features=1, max_depth=6 ..................
            #     # [CV] ... n_estimators=1000, max_features=1, max_depth=6, total=53.5min
            #     # [CV] n_estimators=1000, max_features=1, max_depth=6 ..................
            
            
            
            
            
                
                
            #     # https://datascience.stackexchange.com/questions/74253/what-needs-to-be-done-to-make-n-jobs-work-properly-on-sklearn-in-particular-on
            #     # joblib is optomized for numpy arrays
            #     # so changing pandas to numpy here
                
                
                
                
                
                
            #     # also, here is information from the loky documentation:
            #     # backend: str, ParallelBackendBase instance or None, default: ‘loky’
            #     # Specify the parallelization backend implementation. Supported backends are:
            #     #     “loky” used by default, can induce some communication and memory overhead when exchanging input and output data with the worker Python processes.
            #     #     “multiprocessing” previous process-based backend based on multiprocessing.Pool. Less robust than loky.
            #     #     “threading” is a very low-overhead backend but it suffers from the Python Global Interpreter Lock if the called function relies a lot on Python objects. “threading” is mostly useful when the execution bottleneck is a compiled extension that explicitly releases the GIL (for instance a Cython loop wrapped in a “with nogil” block or an expensive call to a library such as NumPy).
            #     #     finally, you can register backends by calling register_parallel_backend. This will allow you to implement a backend of your liking.
            #     # ...
                
                
            #     # this still isn't doing it, so looking into scikit-learn docs to find out more 
            #     # https://scikit-learn.org/stable/_downloads/scikit-learn-docs.pdf
            #     # ok this is actually really helpful:
            #     # "Joblib is able to support both multi-processing and multi-threading. Whether joblib chooses to spawn a thread or a
            #     # process depends on the backend that it’s using.
            #     # Scikit-learn generally relies on the loky backend, which is joblib’s default backend. Loky is a multi-processing backend. When doing multi-processing, in order to avoid duplicating the memory in each process (which isn’t reasonable
            #     # with big datasets), joblib will create a memmap that all processes can share, when the data is bigger than 1MB.
            #     # In some specific cases (when the code that is run in parallel releases the GIL), scikit-learn will indicate to joblib
            #     # that a multi-threading backend is preferable.
            #     # As a user, you may control the backend that joblib will use (regardless of what scikit-learn recommends) by using a
            #     # context manager:
            #         # from joblib import parallel_backend
            #         # with parallel_backend('threading', n_jobs=2):
            #         # # Your scikit-learn code here
            #     # "
            #     rfc_random = RandomizedSearchCV(estimator = rfc, param_distributions = param_grid, n_iter = 10, cv = 10, verbose=2, random_state=313, n_jobs = 2)
                
            #     from joblib import parallel_backend
            #     with parallel_backend('threading', n_jobs=2):
            #         rfc_random.fit(X_train.to_numpy(), y_train.to_numpy())
            
            #     # Fitting 10 folds for each of 10 candidates, totalling 100 fits
            #     # [CV] n_estimators=1000, max_features=1, max_depth=6 ..................[CV] n_estimators=1000, max_features=1, max_depth=6 ..................
            #     # [Parallel(n_jobs=2)]: Using backend ThreadingBackend with 2 concurrent workers.
            
            #     # oversubscription:
            #     # "It is generally recommended to avoid using significantly more processes or threads than the number of CPUs on a
            #     # machine. Over-subscription happens when a program is running too many threads at the same time.
            #     # Suppose you have a machine with 8 CPUs. Consider a case where you’re running a GridSearchCV (parallelized
            #     # with joblib) with n_jobs=8 over a HistGradientBoostingClassifier (parallelized with OpenMP). Each
            #     # instance of HistGradientBoostingClassifier will spawn 8 threads (since you have 8 CPUs). That’s a total
            #     # of 8 * 8 = 64 threads, which leads to oversubscription of physical CPU resources and to scheduling overhead.
            #     # Oversubscription can arise in the exact same fashion with parallelized routines from MKL, OpenBLAS or BLIS that
            #     # are nested in joblib calls.
            #     # Starting from joblib >= 0.14, when the loky backend is used (which is the default), joblib will tell its child
            #     # processes to limit the number of threads they can use, so as to avoid oversubscription. In practice the heuristic
            #     # that joblib uses is to tell the processes to use max_threads = n_cpus // n_jobs, via their corresponding
            #     # environment variable. Back to our example from above, since the joblib backend of GridSearchCV is loky, each
            #     # process will only be able to use 1 thread instead of 8, thus mitigating the oversubscription issue.
            #     # "
            #     #...so maybe using n_jobs=-1 isn't the best idea for GridSearchCV after all...
            #     #maybe i'm causing oversubscription and actually making the process less efficient
            #     #than if i just specified the n_jobs to be 2 or 4 or something...
                
                
                
                
                
                
            #     # another idea entirely:
            #     #https://stats.stackexchange.com/questions/403749/randomized-search-on-big-dataset
            #     # However, in general there are more optimised solutions for this, i.a. 
            #     # bayesian optimization (check out this great blog post with python code:
            #     # https://thuijskens.github.io/2016/12/29/bayesian-optimisation/). 
            #     # Instead of selecting hyperparameters randomly without any strategy, bayesian 
            #     # optimization tries to find hyperparameters that lead to better results than 
            #     # in the last setting. You approach a better solution step-for-step and probably
            #     # in less time.
                
            #     # This algorithm works well enough, if we can get samples from f cheaply. 
            #     # However, when you are training sophisticated models on large data sets, 
            #     # it can sometimes take on the order of hours, or maybe even days, to get a 
            #     # single sample from f. In those cases, can we do any better than random 
            #     # search? It seems that we should be able to use past samples of f, to determine 
            #     # for which values of x we are going to sample f next.
                
            #     # python implementation of the bayesian gaussian optimization
            #     #https://github.com/thuijskens/bayesian-optimization
                
                
                
            #     #https://towardsdatascience.com/5x-faster-scikit-learn-parameter-tuning-in-5-lines-of-code-be6bdd21833c
            #     # this article gives an implementation of a faster alternative to GridSearchCV and RandomizedSearchCV
            #     # "In this blog post, we introduce tune-sklearn. Tune-sklearn is a drop-in replacement 
            #     # for Scikit-Learn’s model selection module with cutting edge hyperparameter tuning 
            #     # techniques (bayesian optimization, early stopping, distributed execution) — these 
            #     # techniques provide significant speedups over grid search and random search!
            #     # "
            #     # https://github.com/ray-project/tune-sklearn
            #     # https://docs.ray.io/en/latest/installation.html
            #     pip install ray
            #     pip install -U ray  # also recommended: ray[debug]
            
                
            #     # https://towardsdatascience.com/5x-faster-scikit-learn-parameter-tuning-in-5-lines-of-code-be6bdd21833c
            #     pip install tune-sklearn "ray[tune]"
            #     # from sklearn.model_selection import GridSearchCV
            #     from tune_sklearn import TuneGridSearchCV   
                
            #     # rfc_random = RandomizedSearchCV(
            #     #     estimator = rfc, 
            #     #     param_distributions = param_grid, 
            #     #     n_iter = 10, 
            #     #     cv = 10, 
            #     #     verbose=2, 
            #     #     random_state=313, 
            #     #     n_jobs = 2
            #     #     )
            #     # rfc_random.fit(X_train, y_train)
                
                
                
            #     # n_estimators = [10, 50, 100, 500, 1000]
            #     n_estimators = [10, 50, 100] #going to make even less
                
            #     # number of features at every split
            #     # just going to use default='auto' for now
            #     # max_features = ['auto', 'sqrt','log2', np.random.randint(1,4)] #default is auto in sklearn 
            #     max_depth = [2,3,4]
               
            #     # create grid of these parameters that will be passed into the searchCV function
            #     param_grid = {
            #      'n_estimators': n_estimators,
            #      # 'max_features': max_features,
            #      'max_depth': max_depth
            #      }
                
                
                
                
                
            #     tune_search = TuneGridSearchCV(
            #         rfc, 
            #         param_grid, 
            #         # n_iter = 10, 
            #         # cv = 10, 
            #         # verbose=2, 
            #         # random_state=313, 
            #         # n_jobs = 2
                    
            #         early_stopping="MedianStoppingRule",
            #         max_iters=10 #,
            #         # global_checkpoint_period=np.inf #https://github.com/ray-project/ray/issues/7718
            #     )
            #     # rfc_tunegrid.fit(X_train, y_train)
                
            #     import time # Just to compare fit times
            #     start = time.time()
            #     tune_search.fit(X_train, y_train)
            #     end = time.time()
            #     print("Tune Fit Time:", end - start)
            #     pred = tune_search.predict(X_test)
            #     accuracy = np.count_nonzero(np.array(pred) == np.array(y_test)) / len(pred)
            #     print("Tune Accuracy:", accuracy)
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            # #ok forget about all of this madness
            # #let's just run it with LinearSVC
            
            # from sklearn.svm import LinearSVC
            # from sklearn.pipeline import make_pipeline
            # from sklearn.preprocessing import StandardScaler
            # from sklearn.datasets import make_classification
            # X, y = make_classification(n_features=4, random_state=0)
            # clf = make_pipeline(StandardScaler(),
            #                      LinearSVC(random_state=0, tol=1e-5))
            # clf.fit(X, y)
            # # Pipeline(steps=[('standardscaler', StandardScaler()),
            # #                 ('linearsvc', LinearSVC(random_state=0, tol=1e-05))])
            
            


# now we can plug these back into the model to see if it improved our performance
def rf_optimal(X_train_rf_optimal, X_test_rf_optimal, y_train_rf_optimal, y_test_rf_optimal):
    rfc_optimal = RandomForestClassifier(
        n_estimators=optimal_n_estimators,
        max_depth=optimal_max_depth, 
        min_samples_split=optimal_min_samples_split, 
        random_state=313, 
        n_jobs=-1)
    rfc_optimal.fit(X_train_rf_optimal, y_train_rf_optimal)
    rfc_optimal_predict = rfc_optimal.predict(X_test_rf_optimal)
    rfc_optimal_cv_score = cross_val_score(rfc_optimal, X_train_rf_optimal, y_train_rf_optimal, cv=10, scoring='roc_auc', n_jobs=-1)
    for i in range(len(rfc_optimal_cv_score)):
        print('CV={}: {}'.format(i+1,rfc_optimal_cv_score[i]))
    print('\n')        
    
    print("=== Confusion Matrix ===")
    print(confusion_matrix(y_test_rf_optimal, rfc_optimal_predict))
    print('\n')
    # === Confusion Matrix ===
    # [[15289    18]
    #  [   25   479]]
    
    print("=== Classification Report ===")
    print(classification_report(y_test_rf_optimal, rfc_optimal_predict))
    print('\n')
    
    print("=== All AUC Scores ===")
    print(rfc_optimal_cv_score)
    print('\n')
    
    print("=== Mean AUC Score ===")
    print("Mean AUC Score - Random Forest: ", rfc_optimal_cv_score.mean())
    
    return rfc_optimal_cv_score

rf_opimal_cv_score = rf_optimal(X_train_scaled, X_test_scaled, y_train, y_test)

# CV=1: 0.9999999256604689
# CV=2: 1.0
# CV=3: 0.9999999203505024
# CV=4: 0.999999378405554
# CV=5: 0.9999999973436134
# CV=6: 0.9970048166131686
# CV=7: 0.9913169963260051
# CV=8: 0.9864830977062352
# CV=9: 0.995122495131181
# CV=10: 0.9970546702323377


# === Confusion Matrix ===
# [[61212     7]
#  [  589  1433]]


# === Classification Report ===
#               precision    recall  f1-score   support

#            0       0.99      1.00      1.00     61219
#            1       1.00      0.71      0.83      2022

#     accuracy                           0.99     63241
#    macro avg       0.99      0.85      0.91     63241
# weighted avg       0.99      0.99      0.99     63241



# === All AUC Scores ===
# [0.99999993 1.         0.99999992 0.99999938 1.         0.99700482
#  0.991317   0.9864831  0.9951225  0.99705467]


# === Mean AUC Score ===
# Mean AUC Score - Random Forest:  0.9966981297769066












#####
# TALK ABOUT MODEL DRIFT
# the error that comes along with using a model fit on one dataset to
# then use new data on that same model...because that new data might 
# not fit the same distributional set 

# bayesian ideology would partially solve for this
# but the problem is that using bayesian modeling on a downsampled dataset
# and then applying it to a larger dataset literally goes directly against 
# the idea of bayesian statistics: that your assumptions should change
# when new data is in play...




# do the KFold before scaling













### Support Vector Machines

### SVM

# Now we can create the SVM model for filling out the table
## Bagging:
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
clf_svm_pred, clf_svm_probs, clf_svm = svm(X_train_scaled, X_test_scaled, y_train, y_test)


# i let this run for over 2 hours and it didn't converge


#########

# https://scikit-optimize.github.io/stable/auto_examples/sklearn-gridsearchcv-replacement.html
# potentially use this to speed it up?

# conda install -c conda-forge scikit-optimize
from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer
from skopt.plots import plot_objective, plot_histogram

from sklearn.datasets import load_digits
from sklearn.svm import LinearSVC, SVC
from sklearn.pipeline import Pipeline


# X, y = load_digits(n_class=10, return_X_y=True)
# X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

# pipeline class is used as estimator to enable
# search over different model types
pipe = Pipeline([
    ('model', SVC())
])

# single categorical value of 'model' parameter is
# sets the model class
# We will get ConvergenceWarnings because the problem is not well-conditioned.
# But that's fine, this is just an example.
linsvc_search = {
    'model': [LinearSVC(max_iter=1000)],
    'model__C': (1e-6, 1e+6, 'log-uniform'),
}

# explicit dimension classes can be specified like this
svc_search = {
    'model': Categorical([SVC()]),
    'model__C': Real(1e-6, 1e+6, prior='log-uniform'),
    'model__gamma': Real(1e-6, 1e+1, prior='log-uniform'),
    'model__degree': Integer(1,8),
    'model__kernel': Categorical(['linear', 'poly', 'rbf']),
}

opt = BayesSearchCV(
    pipe,
    # (parameter space, # of evaluations)
    [(svc_search, 40), (linsvc_search, 16)],
    cv=3
)

opt.fit(X_train_scaled, y_train)

print("val. score: %s" % opt.best_score_)
print("test score: %s" % opt.score(X_test_scaled, y_test))
print("best params: %s" % str(opt.best_params_))


#########


# i believe that this is the best solution i have going right now
# and maybe i'm just too impatient
# but after letting it run for 14 hours (overnight), i gave up

# i'm opting for the option of downsampling my data
# and then using the model fit determined in my downsampled data
# applied to the full 2mil dataset 

#///

#####
# TALK ABOUT MODEL DRIFT
# the error that comes along with using a model fit on one dataset to
# then use new data on that same model...because that new data might 
# not fit the same distributional set 

# bayesian ideology would partially solve for this
# but the problem is that using bayesian modeling on a downsampled dataset
# and then applying it to a larger dataset literally goes directly against 
# the idea of bayesian statistics: that your assumptions should change
# when new data is in play...




# do the KFold before scaling






# http://www.gm.fh-koeln.de/~konen/Publikationen/kochGMA2013.pdf
# my main question here is whether or not there will be degredation in the
# predictive capacity of our model if we use a subsample of the original 
# data that is a substantially small percentage to the original (consider
# something like 1000 observations from the 2mil dataset)...? 
# this paper in section 3 page 9 tests different percentages (10, 25,50) and 
# compares them to the whole, to see how well they perform. the authors report
# only a slight degredation alongside a substantial decrease in the 
# computation time for the ... this is great news! the time complexity of 
# running SVM on a large dataset is too taxing on my personal computer, so 
# knowing that i can have confidence in the findings from my downsampled 
# data (even if i'm using a very small percentage of the original data) is 
# quite a relief 
# i am doing this project off of my laptop, which admittedly is pretty good,
# but i'm still only working with 6 cores and 32gb or RAM, so i can't afford
# to have a massively time/data intensive SVM run on my computer... i need
# speed, i need to be able to iterate quickly and determine the best model
# from the resources that i have
# so downsampling my data must be the path forward since it offers me a "good
# enough" solution along with much improved computation times






len(pixels) #63241
pixels.head()

## Downsample the data 
# 63241 samples will take a long time for the support vector machine to run
# so downsampling will make this process quicker...
# Support Vector Machines are great with small datasets, but not awesome with large ones, 
# and this dataset is big enough to take a long time to optimize with Cross Validation.
# So we'll downsample both categories, Blue Tarp and Non Blue Tarp to 1,000 each.
    # - rationale taken from statquest support vector machine video

def downsample_data(num_samples):    
    #to ensure that i actually get 1000 samples for each category, i'll start by 
    #splitting the data into two dataframes, one for Class=1 (blue tarp) and one 
    #for Class=0 (not blue tarp)
    holdout_blue_tarp = holdout[y_holdout==1]
    holdout_not_blue_tarp = holdout[y_holdout==0]
    
    #let's see how this breaks down
    holdout_blue_tarp.shape #(2022, 4) ... 2022 observations 
    holdout_not_blue_tarp.shape #(61219, 4) ... 61219 observations
    sum(y)/len(y) #0.03197292895431761
    
    
    
    X_holdout, y_holdout
    
    
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






#split this data using the load_data() from above and pass it the downsampled data
X_downsampled, y_downsampled = load_data(pixels_downsampled)
X_downsampled, y_downsampled

X_train_downsampled, X_test_downsampled, y_train_downsampled, y_test_downsampled = kfold_train_test_split(X_downsampled, y_downsampled)

X_train_downsampled_scaled, X_test_downsampled_scaled = scale_X_data(X_train_downsampled, X_test_downsampled)
X_train_downsampled_scaled, X_test_downsampled_scaled

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

#unscaled data
build_svm(X_train_downsampled, X_test_downsampled, y_train_downsampled, y_test_downsampled)
#scaled data
build_svm(X_train_downsampled_scaled,X_test_downsampled_scaled,y_train_downsampled, y_test_downsampled)

#This performs incredibly well, which always makes me pause and think whether I might be overfitting. Are these tests really performing so well or is this just evidence of random forest and SVMs being better solutions for this kind of classification problem than the methods used in Part 1?
#The confusmion matrix shows us that of the 264 observations that were not blue tarp, 258 (97.7%) were correctly classified. Of the 236 observatiosn that were blue tarps, 235 (99.6%) were correctly classified this goes to show how great SVM is as an out-of-the-box solution / machine learning technique.
#I will use cross validation to optimize the parameters to see if i can improve predictions at all but honestly, these numbers are already fantastic. My next step would be to run the entire dataset (without downsampling) to see how it performs.

### SVM -- now we can actually build the SVM model
# use the svm() function I defined above
clf_svm_pred_downsampled, clf_svm_probs_downsampled, clf_svm_downsampled = svm(X_train_downsampled_scaled, X_test_downsampled_scaled, y_train_downsampled, y_test_downsampled)

# When we are optimizing a support vector machine we are attempting to find the best value for gamma and the regularization parameter C that will improve the prediction / classification accuracy of the model. Since we are looking to optimize two parameters (gamma and C) we will use GridSearchCV(), specifying several values for gamma and C and then letting the function determine the optimal combination (that's the great thing about using GridSerachCV() for problems like this: it tests all possible combinations of parameters for us and all we have to do is plug in some options).
# The default values for the SVC parameters in sklearn.svm are:
# C=1.0
# kernel='rbf
# degree=3
# gamma='scale' (which is 1/(n_features*X.var()) while 'auto' uses 1/n_features
# GridSearchSV in sklearn.model_selection has a parameter called param_grid where you can pass in a dictionary of parameters. I will pass in all of the default SVC parameters and then some extra parameters so that the GridSearchCV() can find the optimal combination from many options.
# My assumption is that 'rbf' will work the best for the pixel data, so I might not need to have all these options for kernel, but I am including 'linear','poly', and 'sigmoid' anyway.


def optimizing_svm(): 
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
print(optimal_params)

optimal_params.fit(X_train_downsampled_scaled, y_train_downsampled)


# Save the optimal parameters down in their own varaibles
best_C = optimal_params.best_params_['C']
best_degree = optimal_params.best_params_['degree']
best_gamma = optimal_params.best_params_['gamma']
best_kernel = optimal_params.best_params_['kernel']

# Then print out the optimal parameters for the model
print('''The best cost for clf_svm is: {}
the best degree for clf_svm is: {}
the best gamma for clf_svm is: {}
the best kernel for clf_svm is: '{}'
      '''.format(
      best_C,
      best_degree,
      best_gamma,
      best_kernel
)) 
    
    
# As expected, the best kernel for the SVM model is 'rbf'.
# Now we can use these optimal parameters to fit a new model:
clf_svm_optimal = SVC(random_state=313, C=best_C, degree=best_degree, gamma=best_gamma, kernel=best_kernel)

build_svm(X_train_downsampled_scaled, X_test_downsampled_scaled, y_train_downsampled, y_test_downsampled, clf_svm = clf_svm_optimal)
    
    
    
    







Completing Table 2

#scaled data
knn_pred_scaled, knn_probs_scaled = knn(X_train_scaled, X_test_scaled, y_train, y_test)
lda_pred_scaled, lda_probs_scaled = lda(X_train_scaled, X_test_scaled, y_train, y_test)
#scaled data
qda_pred_scaled, qda_probs_scaled = qda(X_train_scaled, X_test_scaled, y_train, y_test)
#scaled data
logreg_pred_scaled, logreg_probs_scaled = logistic_regression(X_train_scaled, X_test_scaled, y_train, y_test)
# scaled data
bagging_pred_scaled, bagging_probs_scaled, rfc_scaled = bagging(X_train_scaled, X_test_scaled, y_train, y_test)
#scaled data
eval_perform_rf(rfc_scaled, X, y, y_test, bagging_pred_scaled)






Accuracy

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
svm_accuracy = accuracy(y_test_downsampled, clf_svm_pred, 'SVM')





AUC
We can then use the roc_auc_score() function to calculate the true-positive rate and false-positive rate for the predictions using a set of thresholds that can then be used to create a ROC Curve plot.
def calculate_AUC(y_test, prob):
    # calculate scores
    auc = roc_auc_score(y_test, prob)
    return auc

auc_KNN = calculate_AUC(y_test, knn_pred)  
print('auc_KNN:', auc_KNN)
auc_LDA = calculate_AUC(y_test, lda_pred)  
print('auc_QDA:',auc_LDA)
auc_QDA = calculate_AUC(y_test, qda_pred)  
print('auc_QDA:', auc_QDA)
auc_LogisticRegression = calculate_AUC(y_test, logreg_pred) 
print('auc_LogisticRegression:',auc_LogisticRegression)
auc_Bagging = calculate_AUC(y_test, bagging_pred) 
print('auc_Bagging:',auc_Bagging)
auc_SVM = calculate_AUC(y_test_downsampled, clf_svm_pred) 
print('auc_SVM:',auc_SVM)








Threshold for ROC
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







ROC
def calculate_ROC(y_test, prob, Type):    
    # calculate roc curves
    fpr, tpr, thresholds = roc_curve(y_test, prob)
    
    # I cannot figure out why there are so fewer threshold values for KNN than for the other models...
    #print(thresholds)
    #fpr, tpr, threshold = roc_curve(y_test, knn_probs_scaled)
    #roc_auc = auc(fpr, tpr)
    #roc_auc
    
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


There's something strange going on with the knn_probs. I was having problems with my KNN graph in Part 1 also, and I still don't know what's going on.


roc_KNN = calculate_ROC(y_test, knn_probs_scaled, Type='K-Nearest Neighbors') 
roc_LDA = calculate_ROC(y_test, lda_probs, Type='LDA')  
roc_QDA = calculate_ROC(y_test, qda_probs, Type='QDA') 
roc_LogisticRegression = calculate_ROC(y_test, logreg_probs,Type='Logistic Regression') 
roc_Bagging = calculate_ROC(y_test, bagging_probs,Type='Bagging') 
roc_SVM = calculate_ROC(y_test, clf_svm_probs,Type='SVM')







Confusion Matrix
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
knn_sensitivity = 0.9746192893 # 192/(192+5) TPR = Sensitivity = TP/(TP+FN) ... True Positive Rate (or Recall or Sensi
knn_specificity = 0.999510364 #1-0.0016332396942575292 Specificity = 1 - FPR = TN/(TN+FP) = 6124/(6124+ 3)
knn_fpr = 0.000489636 # 3/(6124+3) = FPR = 1 - Specificity = FP/(TN+FP) ... False Positive Rate defines how many incorrect positive results occur among all negative samples available during the test.
knn_precision = 0.9846153846 # 192/(192+3) Precision = TruePositives / (TruePositives + FalsePositives)


#LDA
lda_confusion_matrix = conf_m(y_test,lda_pred_scaled)
lda_sensitivity = 0.7969543147 # 157/(157+40) TPR = Sensitivity = TP/(TP+FN) ... True Positive Rate (or Recall or Sensi
lda_specificity = 0.9890647952 #1-0.010844711569869995 = Specificity = 1 - FPR = TN/(TN+FP) = 6060/(6060+67) 
lda_fpr = 0.01093520483 # 67/(6060+67) = FPR = 1 - Specificity = FP/(TN+FP) ... False Positive Rate defines how many incorrect positive results occur among all negative samples available during the test.
lda_precision = 0.7008928571 # 157/(157+67) Precision = TruePositives / (TruePositives + FalsePositives)



#QDA
qda_confusion_matrix = conf_m(y_test,qda_pred)
qda_sensitivity = 0.8730964467 # 172/(172+25) TPR = Sensitivity = TP/(TP+FN) ... True Positive Rate (or Recall or Sensi
qda_specificity = 0.999510364 #1-0.00039197752662180704 = Specificity = 1 - FPR = TN/(TN+FP) = 6124/(6124+3)
qda_fpr = 0.0004896360372 # 6/(15301+6) = FPR = 1 - Specificity = FP/(TN+FP) ... False Positive Rate defines how many incorrect positive results occur among all negative samples available during the test.
qda_precision = 0.9828571429 # 172/(172+3) Precision = TruePositives / (TruePositives + FalsePositives)


#Logistic Regression
logistic_regression_confusion_matrix = conf_m(y_test,logreg_pred)
logreg_sensitivity = 0.923857868 # 182/(182+15) TPR = Sensitivity = TP/(TP+FN) ... True Positive Rate (or Recall or Sensi
logreg_specificity = 0.9990207279 #1-0.0009146142287842164 Specificity = 1 - FPR = TN/(TN+FP) = 6121/(6121+6)
logreg_fpr = 0.0009792720744 # 14/(15293+14) = FPR = 1 - Specificity = FP/(TN+FP) ... False Positive Rate defines how many incorrect positive results occur among all negative samples available during the test.
logreg_precision = 0.9680851064 # 182/(182+6) Precision = TruePositives / (TruePositives + FalsePositives)


#Bagging
#TN FP
#FN T
bagging_confusion_matrix = conf_m(y_test,bagging_pred)
bagging_sensitivity = 0.9471428571428572 # 663/(663+37) TPR = Sensitivity = TP/(TP+FN) ... True Positive Rate (or Recall or Sensi
bagging_specificity = 0.9985126425384234 #1-0.0009146142287842164 Specificity = 1 - FPR = TN/(TN+FP) = 20140/(20140+30)
bagging_fpr = 0.001487357461576599 # 30/(20140+30) = FPR = 1 - Specificity = FP/(TN+FP) ... False Positive Rate defines how many incorrect positive results occur among all negative samples available during the test.
bagging_precision = 0.9567099567099567 # 663/(663+30) Precision = TruePositives / (TruePositives + FalsePositives)


#SVM
svm_confusion_matrix = conf_m(y_test, clf_svm_pred)
svm_sensitivity = 0.9471428571428572 # 663/(663+37) TPR = Sensitivity = TP/(TP+FN) ... True Positive Rate (or Recall or Sensi
svm_specificity = 0.9985126425384234 #1-0.0009146142287842164 Specificity = 1 - FPR = TN/(TN+FP) = 20140/(20140+30)
svm_fpr = 0.001487357461576599 # 30/(20140+30) = FPR = 1 - Specificity = FP/(TN+FP) ... False Positive Rate defines how many incorrect positive results occur among all negative samples available during the test.
svm_precision = 0.9567099567099567 # 663/(663+30) Precision = TruePositives / (TruePositives + FalsePositives)







0          61214     5
1            611  1411
189    0
1            3  208
tn = 189
fp = 0
fn = 3
tp = 208

sensitivity = tp / (tp + fn) 
specificity = tn / (tn + fp)
fpr = 1-specificity
precision = tp / (tp + fp)

print('sensitivity:',sensitivity)
print('specificity:',specificity)
print('fpr:',fpr)
print('precision:',precision)




