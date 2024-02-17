#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 16 16:02:44 2024

@author: Jeremy Chen
"""

#first part is the preprocessing of the cloud dataset
import pandas as pd
import numpy as np
import copy
import math

data_entire=pd.read_csv('dataset/cloud.data',header=None,error_bad_lines=False)
data_db1 = data_entire.iloc[37:1061]
data_db2 = data_entire.iloc[1064:-3]

#transfer original datatype from str to float
#for subset 1 and subset 2

def data_transfer(data_df):
    new_data_ar = np.zeros((len(data_df), 10))
    for i in range(len(data_df)):
        de = np.array(data_df.iloc[i])
        de_tr = de[0].split()
        de_trnew = [float(e) for e in de_tr]
        #de_tr[e] = float(de_tr[e]) for e in range(len(de_tr))
        new_data_ar[i,:] = de_trnew
    return new_data_ar

#define the function for standardization and inverse_transform
#data_new is numpy array form
import copy
def transform(new_data_ar):
    #deepcopy another entity to avoid change original entity
    data_new = copy.deepcopy(new_data_ar)
    for i in range(data_new.shape[1]):
        d_mean = np.mean(data_new[:,i])
        d_std = np.std(data_new[:,i])
        data_new[:,i] = (data_new[:,i]-d_mean)/d_std
    return data_new

def inverse_transform(data_new, new_data_ar):
    data_back = copy.deepcopy(data_new)
    for i in range(data_new.shape[1]):
        d_mean = np.mean(new_data_ar[:,i])
        d_std = np.std(new_data_ar[:,i])
        data_back[:,i] = (data_new[:,i]*d_std) + d_mean
    return data_back

#select the former 9 dims data as features and the last column as label
#import existing ML models from sklearn package, treat them as experts
import matplotlib.pyplot as plt
import seaborn as sns

from pylab import mpl
mpl.rcParams['font.sans-serif'] = ['SimHei']

from matplotlib import rcParams
rcParams['axes.unicode_minus']=False
 
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

#import models

#Linear Regression
from sklearn import linear_model
model_LinearRegression = linear_model.LinearRegression()
#Decision Tree Regressor
from sklearn import tree
model_DecisionTreeRegressor = tree.DecisionTreeRegressor()
#SVM Regressor
from sklearn import svm
model_SVR = svm.SVR()
#K Neighbors Regressor
from sklearn import neighbors
model_KNeighborsRegressor = neighbors.KNeighborsRegressor()
#Random Forest Regressor
from sklearn import ensemble
model_RandomForestRegressor = ensemble.RandomForestRegressor(n_estimators=20)
#Adaboost Regressor
from sklearn import ensemble
model_AdaBoostRegressor = ensemble.AdaBoostRegressor(n_estimators=50)

#the updating mechanism of the static expert
#p_i_t should be a list, with the length of total test set
#set a numpy array to store all p_i_t, size = (len(test_set), 6)

def generate_pit(data_test):
    pit_series = np.zeros((data_test.shape[0], 6))
    pit_init = np.random.uniform(0,1,size=6)
    pit_series[0,:] = pit_init
    return pit_series

#define the function for updating the pit
#loss should be a array contains loss of each expert in each timestep, size = [t,6]
def update_pit_static(pit_series, t, loss, Zt, yita, alpha):
    for i in range(loss.shape[1]):
        pit_series[t+1,i] = pit_series[t,i]*math.exp(-yita*loss[t,i])*Zt
    return pit_series

#define the function for updating the pit
#loss should be a array contains loss of each expert in each timestep, size = [t,6]
#for fixed alpha, we should define more entities
def update_pit_fixed(pit_series, t, loss, Zt, yita, alpha):
    for i in range(loss.shape[1]):
        for j in range(loss.shape[1]):
            theta_ij = 1-alpha if j == i else alpha/5
            pit_series[t+1,i] += pit_series[t,j]*math.exp(-yita*loss[t,j])*Zt*theta_ij
                
    return pit_series

#distribute the pi weights among experts and generate the weighted predictions of learner

def main_func(data_db, expert_fix, alpha):
    #firstly data standardization
    new_data_ar = data_transfer(data_db)
    data_new = transform(new_data_ar)
    X_train = data_new[:7*128, :-1]
    Y_train = data_new[:7*128, -1:]
    X_test = data_new[7*128:, :-1]
    Y_test = data_new[7*128:, -1:]
    
    #initialize weights
    pit_series = generate_pit(Y_test)
    #select the value for Zt as 0.2 randomly
    Zt = 1
    #yita value is defined as e-3
    yita = 0.1
    #build a loss store array
    loss_learner = np.zeros((Y_test.shape[0], 1))
    loss_expert = np.zeros((Y_test.shape[0], 6))
    #build prediction array for all experts
    pred_experts = np.zeros((Y_test.shape[0], 6))
    #build prediction array of the learner
    pred_learner = np.zeros((Y_test.shape[0], 1))
    
    
    #fit all models(experts)
    ##Linear Regression
    model_LinearRegression.fit(X_train,Y_train)
    #Decision Tree Regressor
    model_DecisionTreeRegressor.fit(X_train,Y_train)
    #SVM Regressor
    model_SVR.fit(X_train,Y_train)
    ##K Neighbors Regressor
    model_KNeighborsRegressor.fit(X_train,Y_train)
    #Random Forest Regressor
    model_RandomForestRegressor.fit(X_train,Y_train)
    #Adaboost Regressor
    model_AdaBoostRegressor.fit(X_train,Y_train)
    
    # build a loop to predict each point step by step
    for t in range(X_test.shape[0]-1):
        y_pred1 = model_LinearRegression.predict(X_test[t,:].reshape(-1,9))
        y_pred2 = model_DecisionTreeRegressor.predict(X_test[t,:].reshape(-1,9))
        y_pred3 = model_SVR.predict(X_test[t,:].reshape(-1,9))
        y_pred4 = model_KNeighborsRegressor.predict(X_test[t,:].reshape(-1,9))
        y_pred5 = model_RandomForestRegressor.predict(X_test[t,:].reshape(-1,9))
        y_pred6 = model_AdaBoostRegressor.predict(X_test[t,:].reshape(-1,9))
        pred_experts[t,:] = [y_pred1,y_pred2,y_pred3,y_pred4,y_pred5,y_pred6]
        
        #calculate the loss of each expert
        loss_expert[t,0] = 0.5*(y_pred1-Y_test[t,:])**2
        loss_expert[t,1] = 0.5*(y_pred2-Y_test[t,:])**2
        loss_expert[t,2] = 0.5*(y_pred3-Y_test[t,:])**2
        loss_expert[t,3] = 0.5*(y_pred4-Y_test[t,:])**2
        loss_expert[t,4] = 0.5*(y_pred5-Y_test[t,:])**2
        loss_expert[t,5] = 0.5*(y_pred6-Y_test[t,:])**2
        
        #the weighted prediction of the learner
        pred_total = pit_series[t,0]*y_pred1 + pit_series[t,1]*y_pred2 + pit_series[t,2]*y_pred3 + pit_series[t,3]*y_pred4 + pit_series[t,4]*y_pred5 + pit_series[t,5]*y_pred6
        pred_learner[t,:] = pred_total
        #pred_learner[t,0] = pit_series[t,0]*y_pred1
        #print("pred_learner is:", pred_learner)
        loss_learner[t,:] = 0.5*(pred_total-Y_test[t,:])**2
        
        #each timestep, pti should be updated based on Bayesian algorithm 
        if expert_fix == True:
            pit_series = update_pit_fixed(pit_series, t, loss_expert, Zt, yita, alpha)
        else:
            pit_series = update_pit_static(pit_series, t, loss_expert, Zt, yita, alpha)
        
    return pit_series, loss_learner, loss_expert, pred_experts, pred_learner
    
    
#draw the evolution curve of these information    
    
#curve for 6 experts' performance and pitseries

def evolution_obs(series_his, title):
    plt.figure(1)
    fs = (10,4)
    plt.figure(figsize=fs,dpi=120)
    xs = np.arange(series_his.shape[0])
    #xs = np.arange(288*2)
    
    plt.title(title)
    plt.xlabel('Lens')
    plt.ylabel('Labels')
    plt.plot(xs, series_his[:,0], '-.g', linewidth = 0.8, label = 'expert1')
    plt.plot(xs, series_his[:,1], '-.r', linewidth = 0.8, label = 'expert2')
    plt.plot(xs, series_his[:,2], '-.b', linewidth = 0.8, label = 'expert3')
    plt.plot(xs, series_his[:,3], '-.y', linewidth = 0.8, label = 'expert4')
    plt.plot(xs, series_his[:,4], '-.c', linewidth = 0.8, label = 'expert5')
    plt.plot(xs, series_his[:,5], '-.k', linewidth = 0.8, label = 'expert6')
    


    plt.grid(linestyle='-.', linewidth = 1)
    plt.legend()
    #plt.savefig('model_folder.png', dpi=800)
    plt.show()

    
def learner_obs(learner_his, title):
    plt.figure(1)
    fs = (10,4)
    plt.figure(figsize=fs,dpi=120)
    xs = np.arange(learner_his.shape[0])
    #xs = np.arange(288*2)
    
    plt.title(title)
    plt.xlabel('Lens')
    plt.ylabel('Values')
    plt.plot(xs, learner_his[:,0], '-.r', linewidth = 0.8, label = 'learner')
    
    


    plt.grid(linestyle='-.', linewidth = 1)
    plt.legend()
    #plt.savefig('model_folder.png', dpi=800)
    plt.show()
       
        
if __name__ == "__main__":
    
    expert_fix = 0
    alpha = 0.3
    pit_series, loss_learner, loss_expert, pred_experts, pred_learner= main_func(data_db2,expert_fix,alpha)
    #
    # draw evolution of weights
    evolution_obs(pit_series,'Pit evolution')
    evolution_obs(loss_expert, 'Experts loss evolution')
    
    evolution_obs(pred_experts,'Prediction of 6 experts')
    learner_obs(pred_learner, 'Prediction of learner')       
        



