#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 19 10:45:38 2024

@author: Jeremy Chen
"""


#import spambase dataset from the 
from ucimlrepo import fetch_ucirepo 
  
# fetch dataset 
spambase = fetch_ucirepo(id=94) 
  
# data (as pandas dataframes) 
X = spambase.data.features 
Y = spambase.data.targets 

#Using classification models from the sklearn package

from sklearn import svm, tree, neighbors, neural_network
import xgboost
import numpy as np
from sklearn.ensemble import VotingClassifier, GradientBoostingClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.feature_selection import VarianceThreshold
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.naive_bayes import MultinomialNB
import matplotlib.pyplot as plt

from sklearn.gaussian_process.kernels import RBF
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from math import log 
import math

#set 6 experts for classification assignment
model=[
    # SVM
    svm.SVC(C=2, kernel='poly',degree=5, probability=True, decision_function_shape='ovo'),
    # bayes
    MultinomialNB(alpha=0.01),
    # AdaBoost classifier
    AdaBoostClassifier(),
    # mlp
    neural_network.MLPClassifier(alpha=1, max_iter=1000),
    # xgBoost
    xgboost.XGBClassifier(),
    #GBDT
    GradientBoostingClassifier(n_estimators=200)
    
]

#the updating mechanism of the static expert
#p_i_t should be a list, with the length of total test set
#set a numpy array to store all p_i_t, size = (len(test_set), 6)

def generate_pit(data_test):
    pit_series = np.zeros((data_test.shape[0], 6))
    #pit_init = np.random.uniform(0,1,size=6)
    pit_init = [1/6]*6
    pit_series[0,:] = pit_init
    return pit_series


#define the function for updating the pit
#loss should be a array contains loss of each expert in each timestep, size = [t,6]
def update_pit_static(pit_series, t, loss, Zt, yita, alpha):
    for i in range(loss.shape[1]):
        pit_series[t+1,i] = pit_series[t,i]*math.exp(-yita*loss[t,i])*Zt
    pit_series[t+1,:] /= np.sum(pit_series[t+1,:])
    return pit_series


#define the function for updating the pit
#loss should be a array contains loss of each expert in each timestep, size = [t,6]
#for fixed alpha, we should define more entities
def update_pit_fixed(pit_series, t, loss, Zt, yita, alpha):
    for i in range(loss.shape[1]):
        for j in range(loss.shape[1]):
            theta_ij = 1-alpha if j == i else alpha/5
            pit_series[t+1,i] += pit_series[t,j]*math.exp(-yita*loss[t,j])*Zt*theta_ij
    pit_series[t+1,:] /= np.sum(pit_series[t+1,:])           
    return pit_series


# distribute the pi weights among experts and generate the weighted predictions of learner

def main_func(X,Y, expert_fix, alpha, model):
    #firstly data standardization
    x=X.values
    y = Y.values
    index = [i for i in range(len(x))]
    np.random.shuffle(index)
    x = x[index]
    y = y[index]
    X_train, Y_train = x[:2000,:20], y[:2000,:]
    X_test, Y_test =x[4550:,:20], y[4550:,:]
    
    #initialize weights
    pit_series = generate_pit(Y_test)
    #select the value for Zt as 0.2 randomly
    Zt = 1
    #yita value is defined as e-3
    yita = 0.001
    #build a loss store array
    loss_learner = np.zeros((Y_test.shape[0], 1))
    loss_expert = np.zeros((Y_test.shape[0], 6))
    #build prediction array for all experts
    pred_experts = np.zeros((Y_test.shape[0], 6))
    #build prediction array of the learner
    pred_learner = np.zeros((Y_test.shape[0], 1))
    
    
    #fit all models(experts)
    ##svm
    model[0].fit(X_train,Y_train)
    #Naive Bayes classifier
    model[1].fit(X_train,Y_train)
    #AdaBoost classifier
    model[2].fit(X_train,Y_train)
    #mlp
    model[3].fit(X_train,Y_train)
    #random forest
    model[4].fit(X_train,Y_train)
    #GBDT
    model[5].fit(X_train,Y_train)
    
    
    # build a loop to predict each point step by step
    for t in range(X_test.shape[0]-1):
        y_pred1 = model[0].predict_proba(X_test[t,:].reshape(-1,20))[:,1]
        y_pred2 = model[1].predict_proba(X_test[t,:].reshape(-1,20))[:,1]
        y_pred3 = model[2].predict_proba(X_test[t,:].reshape(-1,20))[:,1]
        y_pred4 = model[3].predict_proba(X_test[t,:].reshape(-1,20))[:,1]
        y_pred5 = model[4].predict_proba(X_test[t,:].reshape(-1,20))[:,1]
        y_pred6 = model[5].predict_proba(X_test[t,:].reshape(-1,20))[:,1]
        pred_experts[t,:] = [y_pred1,y_pred2,y_pred3,y_pred4,y_pred5,y_pred6]
        
        #calculate the loss of each expert
        loss_expert[t,0] = -((1-Y_test[t,:])*log(1-y_pred1)+Y_test[t,:]*log(y_pred1))
        loss_expert[t,1] = -((1-Y_test[t,:])*log(1-y_pred2)+Y_test[t,:]*log(y_pred2))
        loss_expert[t,2] = -((1-Y_test[t,:])*log(1-y_pred3)+Y_test[t,:]*log(y_pred3))
        loss_expert[t,3] = -((1-Y_test[t,:])*log(1-y_pred4)+Y_test[t,:]*log(y_pred4))
        loss_expert[t,4] = -((1-Y_test[t,:])*log(1-y_pred5)+Y_test[t,:]*log(y_pred5))
        loss_expert[t,5] = -((1-Y_test[t,:])*log(1-y_pred6)+Y_test[t,:]*log(y_pred6))
        
        #the weighted prediction of the learner
        pred_total = pit_series[t,0]*y_pred1 + pit_series[t,1]*y_pred2 + pit_series[t,2]*y_pred3 + pit_series[t,3]*y_pred4 + pit_series[t,4]*y_pred5 + pit_series[t,5]*y_pred6
        pred_learner[t,:] = pred_total
        #pred_learner[t,0] = pit_series[t,0]*y_pred1
        #print("pred_learner is:", pred_learner)
        loss_learner[t,:] = -((1-Y_test[t,:])*log(1-pred_total)+Y_test[t,:]*log(pred_total))
        
        #each timestep, pti should be updated based on Bayesian algorithm 
        if expert_fix == True:
            pit_series = update_pit_fixed(pit_series, t, loss_expert, Zt, yita, alpha)
        else:
            pit_series = update_pit_static(pit_series, t, loss_expert, Zt, yita, alpha)
        
    return pit_series, loss_learner, loss_expert, pred_experts, pred_learner, Y_test


#plot the evolution curve of these information    
    
#curve for 6 experts' performance and pitseries

def evolution_obs(series_his, title):
    plt.figure(1)
    fs = (10,4)
    plt.figure(figsize=fs,dpi=120)
    xs = np.arange(series_his.shape[0])
    #xs = np.arange(288*2)
    
    plt.title(title)
    plt.xlabel('Steps')
    plt.ylabel('Values')
    plt.plot(xs, series_his[:,0], '-.g', linewidth = 0.8, label = 'expert1')
    plt.plot(xs, series_his[:,1], '-.r', linewidth = 0.8, label = 'expert2')
    plt.plot(xs, series_his[:,2], '-.b', linewidth = 0.8, label = 'expert3')
    plt.plot(xs, series_his[:,3], '-.y', linewidth = 0.8, label = 'expert4')
    plt.plot(xs, series_his[:,4], '-.c', linewidth = 0.8, label = 'expert5')
    plt.plot(xs, series_his[:,5], '-.k', linewidth = 0.8, label = 'expert6')
    


    plt.grid(linestyle='-.', linewidth = 1)
    plt.legend()
    plt.savefig(title, dpi=800)
    plt.show()

    
def learner_obs(learner_his, title):
    plt.figure(1)
    fs = (10,4)
    plt.figure(figsize=fs,dpi=120)
    xs = np.arange(learner_his.shape[0])
    #xs = np.arange(288*2)
    
    plt.title(title)
    plt.xlabel('Steps')
    plt.ylabel('Values')
    plt.plot(xs, learner_his[:,0], '-.r', linewidth = 0.8, label = 'learner')
    
    


    plt.grid(linestyle='-.', linewidth = 1)
    plt.legend()
    plt.savefig(title, dpi=800)
    plt.show()


if __name__ == "__main__":
    
    expert_fix = True # decide whether to use fixed-share alpha 
    alpha = 0.8  #can be adjusted to decide the alpha value for fixed expert
    pit_series, loss_learner, loss_expert, pred_experts, pred_learner,Y_test = main_func(X,Y,expert_fix,alpha,model)
    #calculate cumulative loss for learner and experts
    cumuloss_learner = np.cumsum(loss_learner, axis=0)
    cumuloss_expert = np.cumsum(loss_expert, axis=0)
    
    # draw evolution of weights
    evolution_obs(pit_series,'Weights evolution for Spambase')
    #plot evolution of learner's cumulative loss
    learner_obs(cumuloss_learner, 'Learner loss evolution for Spambase')
    evolution_obs(cumuloss_expert, 'Experts loss evolution for Spambase')





















