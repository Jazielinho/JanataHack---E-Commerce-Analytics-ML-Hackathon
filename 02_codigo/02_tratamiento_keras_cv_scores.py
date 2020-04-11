# -*- coding: utf-8 -*-
"""
Created on Fri Apr 10 08:48:33 2020

@author: Jazielinho
"""

import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score

PATH_MODELS = 'D:/Concursos_Kaggle/36_Janatack/03_model/02_keras_cv/'
PATH_SAVE_SCORES = 'D:/Concursos_Kaggle/36_Janatack/03_model/02_keras_cv/'

ID = 'session_id'
TARGET = 'gender'


data_cv_list = [pd.concat([pd.read_csv(PATH_MODELS + 'cv_{}_{}.csv'.format(cv_, x)) for x in range(5)],
                           axis=0) for cv_ in range(10)]
for cv_ in range(10):
    data_cv_list[cv_].set_index('Unnamed: 0', inplace=True)
    data_cv_list[cv_].sort_index(inplace=True)
    
data_cv_np = (data_cv_list[0]['0'].values + data_cv_list[1]['0'].values +
            data_cv_list[2]['0'].values + data_cv_list[3]['0'].values +
            data_cv_list[4]['0'].values + data_cv_list[5]['0'].values +
            data_cv_list[6]['0'].values + data_cv_list[7]['0'].values +
            data_cv_list[8]['0'].values + data_cv_list[9]['0'].values) / 10            


data_cv_df = pd.DataFrame(data_cv_np, columns=['pred_keras_cv'], 
                          index=data_cv_list[0].index)
data_cv_df[TARGET] = data_cv_list[0][TARGET]
data_cv_df.index.name = ID
data_cv_df.to_csv(PATH_SAVE_SCORES + 'cv_keras.csv')


dict_umbral = {}
for i in range(11):
    class_cv = ['male' if x > i /10 else'female' for x in data_cv_df['pred_keras_cv'].values]
    dict_umbral[i/10] = accuracy_score(data_cv_df[TARGET], class_cv)

umbral_optimo = pd.Series(dict_umbral).idxmax()
    



data_test_list = []
for cv_ in range(10):
    data_test_list_ = []
    for cv__ in range(5):
        test_df = pd.read_csv(PATH_MODELS + 'test_{}_{}.csv'.format(cv_, cv__))
        test_df.set_index('Unnamed: 0', inplace=True)
        test_df.sort_index(inplace=True)
        data_test_list_.append(test_df['0'].values)
        list_index_test = test_df.index.tolist()
    data_test_list.append(data_test_list_)
    
data_test_mean_list = []
for cv_ in range(10):
    data_mean = (data_test_list[cv_][0] + data_test_list[cv_][1] +
                 data_test_list[cv_][2] + data_test_list[cv_][3] +
                 data_test_list[cv_][4]) / 5
    data_test_mean_list.append(data_mean)
        
    
data_test_np = (data_test_mean_list[0] + data_test_mean_list[1] +
                data_test_mean_list[2] + data_test_mean_list[3] +
                data_test_mean_list[4] + data_test_mean_list[5] +
                data_test_mean_list[6] + data_test_mean_list[7] +
                data_test_mean_list[8] + data_test_mean_list[9]) / 10            
        
        
data_test_df = pd.DataFrame(data_test_np, columns=['pred_keras_cv'],
                            index=list_index_test)
data_test_df.index.name = ID

data_test_df.to_csv(PATH_SAVE_SCORES + 'test_keras.csv')


data_test_df[TARGET] = data_test_df['pred_keras_cv'].apply(lambda x: 'male' if x > umbral_optimo else 'female')
data_test_df.drop(['pred_keras_cv'], axis=1).to_csv(PATH_SAVE_SCORES + 'test_keras_desicion_optima.csv')

data_test_df[TARGET] = data_test_df['pred_keras_cv'].apply(lambda x: 'male' if x > 0.5 else 'female')
data_test_df.drop(['pred_keras_cv'], axis=1).to_csv(PATH_SAVE_SCORES + 'test_keras_desicion_normal.csv')

