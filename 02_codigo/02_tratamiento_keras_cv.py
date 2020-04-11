# -*- coding: utf-8 -*-
"""
Created on Fri Apr 10 03:53:49 2020

@author: Lookiero
"""

import pandas as pd
import numpy as np
import os
import tqdm

PATH = 'D:/Concursos_Kaggle/36_Janatack/01 datos/'
TRAIN = PATH + 'train_8wry4cB.csv'
TEST = PATH + 'test_Yix80N0.csv'

PATH_MODEL = 'D:/Concursos_Kaggle/36_Janatack/03_model/02_keras_cv/'


BATCH_SIZE = 1024
EPOCH = 5000
NUM_CV = 5
NUM_CV_LIST = 10

ID = 'session_id'
TARGET = 'gender'


data_train = pd.read_csv(TRAIN)
data_test = pd.read_csv(TEST)
data_test[TARGET] = np.nan

data_all = pd.concat([data_train, data_test], axis=0)


data_all.head()


data_all['startTime'] = pd.to_datetime(data_all['startTime'])
data_all['endTime'] = pd.to_datetime(data_all['endTime'])


data_all['dif_time'] = (data_all['endTime'] - data_all['startTime']).apply(lambda x: x.total_seconds())
data_all['num_products'] = data_all['ProductList'].apply(lambda x: len(x.split(';')))
data_all['num_products'].describe()


data_all['target_int'] = data_all[TARGET].apply(lambda x: 1 if x == 'male' else 0)

data_all.boxplot(column='dif_time', by=TARGET)
data_all.boxplot(column='num_products', by=TARGET)



list_products = data_all['ProductList'].apply(lambda x: pd.Series(x.split(';')))
list_products.fillna('/unk_category/unk_sub_category/unq_sub_sub_category/unq_product_final', 
                     inplace=True)


list_category = []
for col in tqdm.tqdm(range(list_products.shape[1])):
    list_products_ = list_products[col]
    list_products_ = list_products_[~list_products_.isnull()]
    list_products_ = list_products_.tolist()
    list_category_ = [x.split('/')[0] for x in list_products_]
    list_category = list_category + list_category_

dict_category = pd.Series(list_category).value_counts().to_dict()
dict_category_int = {k: enum + 1 for enum, k in enumerate(dict_category.keys())}
dict_category_int['unk_category'] = 0


list_sub_category = []
for col in tqdm.tqdm(range(list_products.shape[1])):
    list_products_ = list_products[col]
    list_products_ = list_products_[~list_products_.isnull()]
    list_products_ = list_products_.tolist()
    list_sub_category_ = [x.split('/')[1] for x in list_products_]
    list_sub_category = list_sub_category + list_sub_category_

dict_sub_category = pd.Series(list_sub_category).value_counts().to_dict()
dict_sub_category_int = {k: enum + 1 for enum, k in enumerate(dict_sub_category.keys())}
dict_sub_category_int['unk_sub_category'] = 0


list_sub_sub_category = []
for col in tqdm.tqdm(range(list_products.shape[1])):
    list_products_ = list_products[col]
    list_products_ = list_products_[~list_products_.isnull()]
    list_products_ = list_products_.tolist()
    list_sub_sub_category_ = [x.split('/')[2] for x in list_products_]
    list_sub_sub_category = list_sub_sub_category + list_sub_sub_category_

dict_sub_sub_category = pd.Series(list_sub_sub_category).value_counts().to_dict()
dict_sub_sub_category_int = {k: enum + 1 for enum, k in enumerate(dict_sub_sub_category.keys())}
dict_sub_sub_category_int['unq_sub_sub_category'] = 0


list_product_final = []
for col in tqdm.tqdm(range(list_products.shape[1])):
    list_products_ = list_products[col]
    list_products_ = list_products_[~list_products_.isnull()]
    list_products_ = list_products_.tolist()
    list_product_final_ = [x.split('/')[3] for x in list_products_]
    list_product_final = list_product_final + list_product_final_

dict_product_final = pd.Series(list_product_final).value_counts().to_dict()
dict_product_final_int = {k: enum + 1 for enum, k in enumerate(dict_product_final.keys())}
dict_product_final_int['unq_product_final'] = 0



list_products_1 = list_products[0].apply(lambda x: pd.Series(x.split('/'))).drop(4,axis=1);list_products_1[0] = list_products_1[0].apply(lambda x: dict_category_int[x]);list_products_1[1] = list_products_1[1].apply(lambda x: dict_sub_category_int[x]);list_products_1[2] = list_products_1[2].apply(lambda x: dict_sub_sub_category_int[x]);list_products_1[3] = list_products_1[3].apply(lambda x: dict_product_final_int[x])
list_products_2 = list_products[1].apply(lambda x: pd.Series(x.split('/'))).drop(4,axis=1);list_products_2[0] = list_products_2[0].apply(lambda x: dict_category_int[x]);list_products_2[1] = list_products_2[1].apply(lambda x: dict_sub_category_int[x]);list_products_2[2] = list_products_2[2].apply(lambda x: dict_sub_sub_category_int[x]);list_products_2[3] = list_products_2[3].apply(lambda x: dict_product_final_int[x])
list_products_3 = list_products[2].apply(lambda x: pd.Series(x.split('/'))).drop(4,axis=1);list_products_3[0] = list_products_3[0].apply(lambda x: dict_category_int[x]);list_products_3[1] = list_products_3[1].apply(lambda x: dict_sub_category_int[x]);list_products_3[2] = list_products_3[2].apply(lambda x: dict_sub_sub_category_int[x]);list_products_3[3] = list_products_3[3].apply(lambda x: dict_product_final_int[x])
list_products_4 = list_products[3].apply(lambda x: pd.Series(x.split('/'))).drop(4,axis=1);list_products_4[0] = list_products_4[0].apply(lambda x: dict_category_int[x]);list_products_4[1] = list_products_4[1].apply(lambda x: dict_sub_category_int[x]);list_products_4[2] = list_products_4[2].apply(lambda x: dict_sub_sub_category_int[x]);list_products_4[3] = list_products_4[3].apply(lambda x: dict_product_final_int[x])
list_products_5 = list_products[4].apply(lambda x: pd.Series(x.split('/'))).drop(4,axis=1);list_products_5[0] = list_products_5[0].apply(lambda x: dict_category_int[x]);list_products_5[1] = list_products_5[1].apply(lambda x: dict_sub_category_int[x]);list_products_5[2] = list_products_5[2].apply(lambda x: dict_sub_sub_category_int[x]);list_products_5[3] = list_products_5[3].apply(lambda x: dict_product_final_int[x])
list_products_6 = list_products[5].apply(lambda x: pd.Series(x.split('/'))).drop(4,axis=1);list_products_6[0] = list_products_6[0].apply(lambda x: dict_category_int[x]);list_products_6[1] = list_products_6[1].apply(lambda x: dict_sub_category_int[x]);list_products_6[2] = list_products_6[2].apply(lambda x: dict_sub_sub_category_int[x]);list_products_6[3] = list_products_6[3].apply(lambda x: dict_product_final_int[x])
list_products_7 = list_products[6].apply(lambda x: pd.Series(x.split('/'))).drop(4,axis=1);list_products_7[0] = list_products_7[0].apply(lambda x: dict_category_int[x]);list_products_7[1] = list_products_7[1].apply(lambda x: dict_sub_category_int[x]);list_products_7[2] = list_products_7[2].apply(lambda x: dict_sub_sub_category_int[x]);list_products_7[3] = list_products_7[3].apply(lambda x: dict_product_final_int[x])
list_products_8 = list_products[7].apply(lambda x: pd.Series(x.split('/'))).drop(4,axis=1);list_products_8[0] = list_products_8[0].apply(lambda x: dict_category_int[x]);list_products_8[1] = list_products_8[1].apply(lambda x: dict_sub_category_int[x]);list_products_8[2] = list_products_8[2].apply(lambda x: dict_sub_sub_category_int[x]);list_products_8[3] = list_products_8[3].apply(lambda x: dict_product_final_int[x])
list_products_9 = list_products[8].apply(lambda x: pd.Series(x.split('/'))).drop(4,axis=1);list_products_9[0] = list_products_9[0].apply(lambda x: dict_category_int[x]);list_products_9[1] = list_products_9[1].apply(lambda x: dict_sub_category_int[x]);list_products_9[2] = list_products_9[2].apply(lambda x: dict_sub_sub_category_int[x]);list_products_9[3] = list_products_9[3].apply(lambda x: dict_product_final_int[x])
list_products_10 = list_products[9].apply(lambda x: pd.Series(x.split('/'))).drop(4,axis=1);list_products_10[0] = list_products_10[0].apply(lambda x: dict_category_int[x]);list_products_10[1] = list_products_10[1].apply(lambda x: dict_sub_category_int[x]);list_products_10[2] = list_products_10[2].apply(lambda x: dict_sub_sub_category_int[x]);list_products_10[3] = list_products_10[3].apply(lambda x: dict_product_final_int[x])
list_products_11 = list_products[10].apply(lambda x: pd.Series(x.split('/'))).drop(4,axis=1);list_products_11[0] = list_products_11[0].apply(lambda x: dict_category_int[x]);list_products_11[1] = list_products_11[1].apply(lambda x: dict_sub_category_int[x]);list_products_11[2] = list_products_11[2].apply(lambda x: dict_sub_sub_category_int[x]);list_products_11[3] = list_products_11[3].apply(lambda x: dict_product_final_int[x])
list_products_12 = list_products[11].apply(lambda x: pd.Series(x.split('/'))).drop(4,axis=1);list_products_12[0] = list_products_12[0].apply(lambda x: dict_category_int[x]);list_products_12[1] = list_products_12[1].apply(lambda x: dict_sub_category_int[x]);list_products_12[2] = list_products_12[2].apply(lambda x: dict_sub_sub_category_int[x]);list_products_12[3] = list_products_12[3].apply(lambda x: dict_product_final_int[x])
list_products_13 = list_products[12].apply(lambda x: pd.Series(x.split('/'))).drop(4,axis=1);list_products_13[0] = list_products_13[0].apply(lambda x: dict_category_int[x]);list_products_13[1] = list_products_13[1].apply(lambda x: dict_sub_category_int[x]);list_products_13[2] = list_products_13[2].apply(lambda x: dict_sub_sub_category_int[x]);list_products_13[3] = list_products_13[3].apply(lambda x: dict_product_final_int[x])
list_products_14 = list_products[13].apply(lambda x: pd.Series(x.split('/'))).drop(4,axis=1);list_products_14[0] = list_products_14[0].apply(lambda x: dict_category_int[x]);list_products_14[1] = list_products_14[1].apply(lambda x: dict_sub_category_int[x]);list_products_14[2] = list_products_14[2].apply(lambda x: dict_sub_sub_category_int[x]);list_products_14[3] = list_products_14[3].apply(lambda x: dict_product_final_int[x])
list_products_15 = list_products[14].apply(lambda x: pd.Series(x.split('/'))).drop(4,axis=1);list_products_15[0] = list_products_15[0].apply(lambda x: dict_category_int[x]);list_products_15[1] = list_products_15[1].apply(lambda x: dict_sub_category_int[x]);list_products_15[2] = list_products_15[2].apply(lambda x: dict_sub_sub_category_int[x]);list_products_15[3] = list_products_15[3].apply(lambda x: dict_product_final_int[x])
list_products_16 = list_products[15].apply(lambda x: pd.Series(x.split('/'))).drop(4,axis=1);list_products_16[0] = list_products_16[0].apply(lambda x: dict_category_int[x]);list_products_16[1] = list_products_16[1].apply(lambda x: dict_sub_category_int[x]);list_products_16[2] = list_products_16[2].apply(lambda x: dict_sub_sub_category_int[x]);list_products_16[3] = list_products_16[3].apply(lambda x: dict_product_final_int[x])
list_products_17 = list_products[16].apply(lambda x: pd.Series(x.split('/'))).drop(4,axis=1);list_products_17[0] = list_products_17[0].apply(lambda x: dict_category_int[x]);list_products_17[1] = list_products_17[1].apply(lambda x: dict_sub_category_int[x]);list_products_17[2] = list_products_17[2].apply(lambda x: dict_sub_sub_category_int[x]);list_products_17[3] = list_products_17[3].apply(lambda x: dict_product_final_int[x])
list_products_18 = list_products[17].apply(lambda x: pd.Series(x.split('/'))).drop(4,axis=1);list_products_18[0] = list_products_18[0].apply(lambda x: dict_category_int[x]);list_products_18[1] = list_products_18[1].apply(lambda x: dict_sub_category_int[x]);list_products_18[2] = list_products_18[2].apply(lambda x: dict_sub_sub_category_int[x]);list_products_18[3] = list_products_18[3].apply(lambda x: dict_product_final_int[x])
list_products_19 = list_products[18].apply(lambda x: pd.Series(x.split('/'))).drop(4,axis=1);list_products_19[0] = list_products_19[0].apply(lambda x: dict_category_int[x]);list_products_19[1] = list_products_19[1].apply(lambda x: dict_sub_category_int[x]);list_products_19[2] = list_products_19[2].apply(lambda x: dict_sub_sub_category_int[x]);list_products_19[3] = list_products_19[3].apply(lambda x: dict_product_final_int[x])
list_products_20 = list_products[19].apply(lambda x: pd.Series(x.split('/'))).drop(4,axis=1);list_products_20[0] = list_products_20[0].apply(lambda x: dict_category_int[x]);list_products_20[1] = list_products_20[1].apply(lambda x: dict_sub_category_int[x]);list_products_20[2] = list_products_20[2].apply(lambda x: dict_sub_sub_category_int[x]);list_products_20[3] = list_products_20[3].apply(lambda x: dict_product_final_int[x])
list_products_21 = list_products[20].apply(lambda x: pd.Series(x.split('/'))).drop(4,axis=1);list_products_21[0] = list_products_21[0].apply(lambda x: dict_category_int[x]);list_products_21[1] = list_products_21[1].apply(lambda x: dict_sub_category_int[x]);list_products_21[2] = list_products_21[2].apply(lambda x: dict_sub_sub_category_int[x]);list_products_21[3] = list_products_21[3].apply(lambda x: dict_product_final_int[x])
list_products_22 = list_products[21].apply(lambda x: pd.Series(x.split('/'))).drop(4,axis=1);list_products_22[0] = list_products_22[0].apply(lambda x: dict_category_int[x]);list_products_22[1] = list_products_22[1].apply(lambda x: dict_sub_category_int[x]);list_products_22[2] = list_products_22[2].apply(lambda x: dict_sub_sub_category_int[x]);list_products_22[3] = list_products_22[3].apply(lambda x: dict_product_final_int[x])
list_products_23 = list_products[22].apply(lambda x: pd.Series(x.split('/'))).drop(4,axis=1);list_products_23[0] = list_products_23[0].apply(lambda x: dict_category_int[x]);list_products_23[1] = list_products_23[1].apply(lambda x: dict_sub_category_int[x]);list_products_23[2] = list_products_23[2].apply(lambda x: dict_sub_sub_category_int[x]);list_products_23[3] = list_products_23[3].apply(lambda x: dict_product_final_int[x])
list_products_24 = list_products[23].apply(lambda x: pd.Series(x.split('/'))).drop(4,axis=1);list_products_24[0] = list_products_24[0].apply(lambda x: dict_category_int[x]);list_products_24[1] = list_products_24[1].apply(lambda x: dict_sub_category_int[x]);list_products_24[2] = list_products_24[2].apply(lambda x: dict_sub_sub_category_int[x]);list_products_24[3] = list_products_24[3].apply(lambda x: dict_product_final_int[x])
list_products_25 = list_products[24].apply(lambda x: pd.Series(x.split('/'))).drop(4,axis=1);list_products_25[0] = list_products_25[0].apply(lambda x: dict_category_int[x]);list_products_25[1] = list_products_25[1].apply(lambda x: dict_sub_category_int[x]);list_products_25[2] = list_products_25[2].apply(lambda x: dict_sub_sub_category_int[x]);list_products_25[3] = list_products_25[3].apply(lambda x: dict_product_final_int[x])
list_products_26 = list_products[25].apply(lambda x: pd.Series(x.split('/'))).drop(4,axis=1);list_products_26[0] = list_products_26[0].apply(lambda x: dict_category_int[x]);list_products_26[1] = list_products_26[1].apply(lambda x: dict_sub_category_int[x]);list_products_26[2] = list_products_26[2].apply(lambda x: dict_sub_sub_category_int[x]);list_products_26[3] = list_products_26[3].apply(lambda x: dict_product_final_int[x])
list_products_27 = list_products[26].apply(lambda x: pd.Series(x.split('/'))).drop(4,axis=1);list_products_27[0] = list_products_27[0].apply(lambda x: dict_category_int[x]);list_products_27[1] = list_products_27[1].apply(lambda x: dict_sub_category_int[x]);list_products_27[2] = list_products_27[2].apply(lambda x: dict_sub_sub_category_int[x]);list_products_27[3] = list_products_27[3].apply(lambda x: dict_product_final_int[x])
list_products_28 = list_products[27].apply(lambda x: pd.Series(x.split('/'))).drop(4,axis=1);list_products_28[0] = list_products_28[0].apply(lambda x: dict_category_int[x]);list_products_28[1] = list_products_28[1].apply(lambda x: dict_sub_category_int[x]);list_products_28[2] = list_products_28[2].apply(lambda x: dict_sub_sub_category_int[x]);list_products_28[3] = list_products_28[3].apply(lambda x: dict_product_final_int[x])
list_products_29 = list_products[28].apply(lambda x: pd.Series(x.split('/'))).drop(4,axis=1);list_products_29[0] = list_products_29[0].apply(lambda x: dict_category_int[x]);list_products_29[1] = list_products_29[1].apply(lambda x: dict_sub_category_int[x]);list_products_29[2] = list_products_29[2].apply(lambda x: dict_sub_sub_category_int[x]);list_products_29[3] = list_products_29[3].apply(lambda x: dict_product_final_int[x])
list_products_30 = list_products[29].apply(lambda x: pd.Series(x.split('/'))).drop(4,axis=1);list_products_30[0] = list_products_30[0].apply(lambda x: dict_category_int[x]);list_products_30[1] = list_products_30[1].apply(lambda x: dict_sub_category_int[x]);list_products_30[2] = list_products_30[2].apply(lambda x: dict_sub_sub_category_int[x]);list_products_30[3] = list_products_30[3].apply(lambda x: dict_product_final_int[x])
list_products_31 = list_products[30].apply(lambda x: pd.Series(x.split('/'))).drop(4,axis=1);list_products_31[0] = list_products_31[0].apply(lambda x: dict_category_int[x]);list_products_31[1] = list_products_31[1].apply(lambda x: dict_sub_category_int[x]);list_products_31[2] = list_products_31[2].apply(lambda x: dict_sub_sub_category_int[x]);list_products_31[3] = list_products_31[3].apply(lambda x: dict_product_final_int[x])
list_products_32 = list_products[31].apply(lambda x: pd.Series(x.split('/'))).drop(4,axis=1);list_products_32[0] = list_products_32[0].apply(lambda x: dict_category_int[x]);list_products_32[1] = list_products_32[1].apply(lambda x: dict_sub_category_int[x]);list_products_32[2] = list_products_32[2].apply(lambda x: dict_sub_sub_category_int[x]);list_products_32[3] = list_products_32[3].apply(lambda x: dict_product_final_int[x])
list_products_33 = list_products[32].apply(lambda x: pd.Series(x.split('/'))).drop(4,axis=1);list_products_33[0] = list_products_33[0].apply(lambda x: dict_category_int[x]);list_products_33[1] = list_products_33[1].apply(lambda x: dict_sub_category_int[x]);list_products_33[2] = list_products_33[2].apply(lambda x: dict_sub_sub_category_int[x]);list_products_33[3] = list_products_33[3].apply(lambda x: dict_product_final_int[x])
list_products_34 = list_products[33].apply(lambda x: pd.Series(x.split('/'))).drop(4,axis=1);list_products_34[0] = list_products_34[0].apply(lambda x: dict_category_int[x]);list_products_34[1] = list_products_34[1].apply(lambda x: dict_sub_category_int[x]);list_products_34[2] = list_products_34[2].apply(lambda x: dict_sub_sub_category_int[x]);list_products_34[3] = list_products_34[3].apply(lambda x: dict_product_final_int[x])
list_products_35 = list_products[34].apply(lambda x: pd.Series(x.split('/'))).drop(4,axis=1);list_products_35[0] = list_products_35[0].apply(lambda x: dict_category_int[x]);list_products_35[1] = list_products_35[1].apply(lambda x: dict_sub_category_int[x]);list_products_35[2] = list_products_35[2].apply(lambda x: dict_sub_sub_category_int[x]);list_products_35[3] = list_products_35[3].apply(lambda x: dict_product_final_int[x])
list_products_36 = list_products[35].apply(lambda x: pd.Series(x.split('/'))).drop(4,axis=1);list_products_36[0] = list_products_36[0].apply(lambda x: dict_category_int[x]);list_products_36[1] = list_products_36[1].apply(lambda x: dict_sub_category_int[x]);list_products_36[2] = list_products_36[2].apply(lambda x: dict_sub_sub_category_int[x]);list_products_36[3] = list_products_36[3].apply(lambda x: dict_product_final_int[x])
list_products_37 = list_products[36].apply(lambda x: pd.Series(x.split('/'))).drop(4,axis=1);list_products_37[0] = list_products_37[0].apply(lambda x: dict_category_int[x]);list_products_37[1] = list_products_37[1].apply(lambda x: dict_sub_category_int[x]);list_products_37[2] = list_products_37[2].apply(lambda x: dict_sub_sub_category_int[x]);list_products_37[3] = list_products_37[3].apply(lambda x: dict_product_final_int[x])
list_products_38 = list_products[37].apply(lambda x: pd.Series(x.split('/'))).drop(4,axis=1);list_products_38[0] = list_products_38[0].apply(lambda x: dict_category_int[x]);list_products_38[1] = list_products_38[1].apply(lambda x: dict_sub_category_int[x]);list_products_38[2] = list_products_38[2].apply(lambda x: dict_sub_sub_category_int[x]);list_products_38[3] = list_products_38[3].apply(lambda x: dict_product_final_int[x])
list_products_39 = list_products[38].apply(lambda x: pd.Series(x.split('/'))).drop(4,axis=1);list_products_39[0] = list_products_39[0].apply(lambda x: dict_category_int[x]);list_products_39[1] = list_products_39[1].apply(lambda x: dict_sub_category_int[x]);list_products_39[2] = list_products_39[2].apply(lambda x: dict_sub_sub_category_int[x]);list_products_39[3] = list_products_39[3].apply(lambda x: dict_product_final_int[x])
list_products_40 = list_products[39].apply(lambda x: pd.Series(x.split('/'))).drop(4,axis=1);list_products_40[0] = list_products_40[0].apply(lambda x: dict_category_int[x]);list_products_40[1] = list_products_40[1].apply(lambda x: dict_sub_category_int[x]);list_products_40[2] = list_products_40[2].apply(lambda x: dict_sub_sub_category_int[x]);list_products_40[3] = list_products_40[3].apply(lambda x: dict_product_final_int[x])
list_products_41 = list_products[40].apply(lambda x: pd.Series(x.split('/'))).drop(4,axis=1);list_products_41[0] = list_products_41[0].apply(lambda x: dict_category_int[x]);list_products_41[1] = list_products_41[1].apply(lambda x: dict_sub_category_int[x]);list_products_41[2] = list_products_41[2].apply(lambda x: dict_sub_sub_category_int[x]);list_products_41[3] = list_products_41[3].apply(lambda x: dict_product_final_int[x])
list_products_42 = list_products[41].apply(lambda x: pd.Series(x.split('/'))).drop(4,axis=1);list_products_42[0] = list_products_42[0].apply(lambda x: dict_category_int[x]);list_products_42[1] = list_products_42[1].apply(lambda x: dict_sub_category_int[x]);list_products_42[2] = list_products_42[2].apply(lambda x: dict_sub_sub_category_int[x]);list_products_42[3] = list_products_42[3].apply(lambda x: dict_product_final_int[x])
list_products_43 = list_products[42].apply(lambda x: pd.Series(x.split('/'))).drop(4,axis=1);list_products_43[0] = list_products_43[0].apply(lambda x: dict_category_int[x]);list_products_43[1] = list_products_43[1].apply(lambda x: dict_sub_category_int[x]);list_products_43[2] = list_products_43[2].apply(lambda x: dict_sub_sub_category_int[x]);list_products_43[3] = list_products_43[3].apply(lambda x: dict_product_final_int[x])



data_train_list = [
        list_products_1[~data_all[TARGET].isnull()][0].values,list_products_1[~data_all[TARGET].isnull()][1].values,list_products_1[~data_all[TARGET].isnull()][2].values,list_products_1[~data_all[TARGET].isnull()][3].values,
        list_products_2[~data_all[TARGET].isnull()][0].values,list_products_2[~data_all[TARGET].isnull()][1].values,list_products_2[~data_all[TARGET].isnull()][2].values,list_products_2[~data_all[TARGET].isnull()][3].values,
        list_products_3[~data_all[TARGET].isnull()][0].values,list_products_3[~data_all[TARGET].isnull()][1].values,list_products_3[~data_all[TARGET].isnull()][2].values,list_products_3[~data_all[TARGET].isnull()][3].values,
        list_products_4[~data_all[TARGET].isnull()][0].values,list_products_4[~data_all[TARGET].isnull()][1].values,list_products_4[~data_all[TARGET].isnull()][2].values,list_products_4[~data_all[TARGET].isnull()][3].values,
        list_products_5[~data_all[TARGET].isnull()][0].values,list_products_5[~data_all[TARGET].isnull()][1].values,list_products_5[~data_all[TARGET].isnull()][2].values,list_products_5[~data_all[TARGET].isnull()][3].values,
        list_products_6[~data_all[TARGET].isnull()][0].values,list_products_6[~data_all[TARGET].isnull()][1].values,list_products_6[~data_all[TARGET].isnull()][2].values,list_products_6[~data_all[TARGET].isnull()][3].values,
        list_products_7[~data_all[TARGET].isnull()][0].values,list_products_7[~data_all[TARGET].isnull()][1].values,list_products_7[~data_all[TARGET].isnull()][2].values,list_products_7[~data_all[TARGET].isnull()][3].values,
        list_products_8[~data_all[TARGET].isnull()][0].values,list_products_8[~data_all[TARGET].isnull()][1].values,list_products_8[~data_all[TARGET].isnull()][2].values,list_products_8[~data_all[TARGET].isnull()][3].values,
        list_products_9[~data_all[TARGET].isnull()][0].values,list_products_9[~data_all[TARGET].isnull()][1].values,list_products_9[~data_all[TARGET].isnull()][2].values,list_products_9[~data_all[TARGET].isnull()][3].values,
        list_products_10[~data_all[TARGET].isnull()][0].values,list_products_10[~data_all[TARGET].isnull()][1].values,list_products_10[~data_all[TARGET].isnull()][2].values,list_products_10[~data_all[TARGET].isnull()][3].values,
        list_products_11[~data_all[TARGET].isnull()][0].values,list_products_11[~data_all[TARGET].isnull()][1].values,list_products_11[~data_all[TARGET].isnull()][2].values,list_products_11[~data_all[TARGET].isnull()][3].values,
        list_products_12[~data_all[TARGET].isnull()][0].values,list_products_12[~data_all[TARGET].isnull()][1].values,list_products_12[~data_all[TARGET].isnull()][2].values,list_products_12[~data_all[TARGET].isnull()][3].values,
        list_products_13[~data_all[TARGET].isnull()][0].values,list_products_13[~data_all[TARGET].isnull()][1].values,list_products_13[~data_all[TARGET].isnull()][2].values,list_products_13[~data_all[TARGET].isnull()][3].values,
        list_products_14[~data_all[TARGET].isnull()][0].values,list_products_14[~data_all[TARGET].isnull()][1].values,list_products_14[~data_all[TARGET].isnull()][2].values,list_products_14[~data_all[TARGET].isnull()][3].values,
        list_products_15[~data_all[TARGET].isnull()][0].values,list_products_15[~data_all[TARGET].isnull()][1].values,list_products_15[~data_all[TARGET].isnull()][2].values,list_products_15[~data_all[TARGET].isnull()][3].values,
        list_products_16[~data_all[TARGET].isnull()][0].values,list_products_16[~data_all[TARGET].isnull()][1].values,list_products_16[~data_all[TARGET].isnull()][2].values,list_products_16[~data_all[TARGET].isnull()][3].values,
        list_products_17[~data_all[TARGET].isnull()][0].values,list_products_17[~data_all[TARGET].isnull()][1].values,list_products_17[~data_all[TARGET].isnull()][2].values,list_products_17[~data_all[TARGET].isnull()][3].values,
        list_products_18[~data_all[TARGET].isnull()][0].values,list_products_18[~data_all[TARGET].isnull()][1].values,list_products_18[~data_all[TARGET].isnull()][2].values,list_products_18[~data_all[TARGET].isnull()][3].values,
        list_products_19[~data_all[TARGET].isnull()][0].values,list_products_19[~data_all[TARGET].isnull()][1].values,list_products_19[~data_all[TARGET].isnull()][2].values,list_products_19[~data_all[TARGET].isnull()][3].values,
        list_products_20[~data_all[TARGET].isnull()][0].values,list_products_20[~data_all[TARGET].isnull()][1].values,list_products_20[~data_all[TARGET].isnull()][2].values,list_products_20[~data_all[TARGET].isnull()][3].values,
        list_products_21[~data_all[TARGET].isnull()][0].values,list_products_21[~data_all[TARGET].isnull()][1].values,list_products_21[~data_all[TARGET].isnull()][2].values,list_products_21[~data_all[TARGET].isnull()][3].values,
        list_products_22[~data_all[TARGET].isnull()][0].values,list_products_22[~data_all[TARGET].isnull()][1].values,list_products_22[~data_all[TARGET].isnull()][2].values,list_products_22[~data_all[TARGET].isnull()][3].values,
        list_products_23[~data_all[TARGET].isnull()][0].values,list_products_23[~data_all[TARGET].isnull()][1].values,list_products_23[~data_all[TARGET].isnull()][2].values,list_products_23[~data_all[TARGET].isnull()][3].values,
        list_products_24[~data_all[TARGET].isnull()][0].values,list_products_24[~data_all[TARGET].isnull()][1].values,list_products_24[~data_all[TARGET].isnull()][2].values,list_products_24[~data_all[TARGET].isnull()][3].values,
        list_products_25[~data_all[TARGET].isnull()][0].values,list_products_25[~data_all[TARGET].isnull()][1].values,list_products_25[~data_all[TARGET].isnull()][2].values,list_products_25[~data_all[TARGET].isnull()][3].values,
        list_products_26[~data_all[TARGET].isnull()][0].values,list_products_26[~data_all[TARGET].isnull()][1].values,list_products_26[~data_all[TARGET].isnull()][2].values,list_products_26[~data_all[TARGET].isnull()][3].values,
        list_products_27[~data_all[TARGET].isnull()][0].values,list_products_27[~data_all[TARGET].isnull()][1].values,list_products_27[~data_all[TARGET].isnull()][2].values,list_products_27[~data_all[TARGET].isnull()][3].values,
        list_products_28[~data_all[TARGET].isnull()][0].values,list_products_28[~data_all[TARGET].isnull()][1].values,list_products_28[~data_all[TARGET].isnull()][2].values,list_products_28[~data_all[TARGET].isnull()][3].values,
        list_products_29[~data_all[TARGET].isnull()][0].values,list_products_29[~data_all[TARGET].isnull()][1].values,list_products_29[~data_all[TARGET].isnull()][2].values,list_products_29[~data_all[TARGET].isnull()][3].values,
        list_products_30[~data_all[TARGET].isnull()][0].values,list_products_30[~data_all[TARGET].isnull()][1].values,list_products_30[~data_all[TARGET].isnull()][2].values,list_products_30[~data_all[TARGET].isnull()][3].values,
        list_products_31[~data_all[TARGET].isnull()][0].values,list_products_31[~data_all[TARGET].isnull()][1].values,list_products_31[~data_all[TARGET].isnull()][2].values,list_products_31[~data_all[TARGET].isnull()][3].values,
        list_products_32[~data_all[TARGET].isnull()][0].values,list_products_32[~data_all[TARGET].isnull()][1].values,list_products_32[~data_all[TARGET].isnull()][2].values,list_products_32[~data_all[TARGET].isnull()][3].values,
        list_products_33[~data_all[TARGET].isnull()][0].values,list_products_33[~data_all[TARGET].isnull()][1].values,list_products_33[~data_all[TARGET].isnull()][2].values,list_products_33[~data_all[TARGET].isnull()][3].values,
        list_products_34[~data_all[TARGET].isnull()][0].values,list_products_34[~data_all[TARGET].isnull()][1].values,list_products_34[~data_all[TARGET].isnull()][2].values,list_products_34[~data_all[TARGET].isnull()][3].values,
        list_products_35[~data_all[TARGET].isnull()][0].values,list_products_35[~data_all[TARGET].isnull()][1].values,list_products_35[~data_all[TARGET].isnull()][2].values,list_products_35[~data_all[TARGET].isnull()][3].values,
        list_products_36[~data_all[TARGET].isnull()][0].values,list_products_36[~data_all[TARGET].isnull()][1].values,list_products_36[~data_all[TARGET].isnull()][2].values,list_products_36[~data_all[TARGET].isnull()][3].values,
        list_products_37[~data_all[TARGET].isnull()][0].values,list_products_37[~data_all[TARGET].isnull()][1].values,list_products_37[~data_all[TARGET].isnull()][2].values,list_products_37[~data_all[TARGET].isnull()][3].values,
        list_products_38[~data_all[TARGET].isnull()][0].values,list_products_38[~data_all[TARGET].isnull()][1].values,list_products_38[~data_all[TARGET].isnull()][2].values,list_products_38[~data_all[TARGET].isnull()][3].values,
        list_products_39[~data_all[TARGET].isnull()][0].values,list_products_39[~data_all[TARGET].isnull()][1].values,list_products_39[~data_all[TARGET].isnull()][2].values,list_products_39[~data_all[TARGET].isnull()][3].values,
        list_products_40[~data_all[TARGET].isnull()][0].values,list_products_40[~data_all[TARGET].isnull()][1].values,list_products_40[~data_all[TARGET].isnull()][2].values,list_products_40[~data_all[TARGET].isnull()][3].values,
        list_products_41[~data_all[TARGET].isnull()][0].values,list_products_41[~data_all[TARGET].isnull()][1].values,list_products_41[~data_all[TARGET].isnull()][2].values,list_products_41[~data_all[TARGET].isnull()][3].values,
        list_products_42[~data_all[TARGET].isnull()][0].values,list_products_42[~data_all[TARGET].isnull()][1].values,list_products_42[~data_all[TARGET].isnull()][2].values,list_products_42[~data_all[TARGET].isnull()][3].values,
        list_products_43[~data_all[TARGET].isnull()][0].values,list_products_43[~data_all[TARGET].isnull()][1].values,list_products_43[~data_all[TARGET].isnull()][2].values,list_products_43[~data_all[TARGET].isnull()][3].values,
        data_all[~data_all[TARGET].isnull()]['dif_time'].values, data_all[~data_all[TARGET].isnull()]['num_products'].values
        ]



data_test_list = [
        list_products_1[data_all[TARGET].isnull()][0].values,list_products_1[data_all[TARGET].isnull()][1].values,list_products_1[data_all[TARGET].isnull()][2].values,list_products_1[data_all[TARGET].isnull()][3].values,
        list_products_2[data_all[TARGET].isnull()][0].values,list_products_2[data_all[TARGET].isnull()][1].values,list_products_2[data_all[TARGET].isnull()][2].values,list_products_2[data_all[TARGET].isnull()][3].values,
        list_products_3[data_all[TARGET].isnull()][0].values,list_products_3[data_all[TARGET].isnull()][1].values,list_products_3[data_all[TARGET].isnull()][2].values,list_products_3[data_all[TARGET].isnull()][3].values,
        list_products_4[data_all[TARGET].isnull()][0].values,list_products_4[data_all[TARGET].isnull()][1].values,list_products_4[data_all[TARGET].isnull()][2].values,list_products_4[data_all[TARGET].isnull()][3].values,
        list_products_5[data_all[TARGET].isnull()][0].values,list_products_5[data_all[TARGET].isnull()][1].values,list_products_5[data_all[TARGET].isnull()][2].values,list_products_5[data_all[TARGET].isnull()][3].values,
        list_products_6[data_all[TARGET].isnull()][0].values,list_products_6[data_all[TARGET].isnull()][1].values,list_products_6[data_all[TARGET].isnull()][2].values,list_products_6[data_all[TARGET].isnull()][3].values,
        list_products_7[data_all[TARGET].isnull()][0].values,list_products_7[data_all[TARGET].isnull()][1].values,list_products_7[data_all[TARGET].isnull()][2].values,list_products_7[data_all[TARGET].isnull()][3].values,
        list_products_8[data_all[TARGET].isnull()][0].values,list_products_8[data_all[TARGET].isnull()][1].values,list_products_8[data_all[TARGET].isnull()][2].values,list_products_8[data_all[TARGET].isnull()][3].values,
        list_products_9[data_all[TARGET].isnull()][0].values,list_products_9[data_all[TARGET].isnull()][1].values,list_products_9[data_all[TARGET].isnull()][2].values,list_products_9[data_all[TARGET].isnull()][3].values,
        list_products_10[data_all[TARGET].isnull()][0].values,list_products_10[data_all[TARGET].isnull()][1].values,list_products_10[data_all[TARGET].isnull()][2].values,list_products_10[data_all[TARGET].isnull()][3].values,
        list_products_11[data_all[TARGET].isnull()][0].values,list_products_11[data_all[TARGET].isnull()][1].values,list_products_11[data_all[TARGET].isnull()][2].values,list_products_11[data_all[TARGET].isnull()][3].values,
        list_products_12[data_all[TARGET].isnull()][0].values,list_products_12[data_all[TARGET].isnull()][1].values,list_products_12[data_all[TARGET].isnull()][2].values,list_products_12[data_all[TARGET].isnull()][3].values,
        list_products_13[data_all[TARGET].isnull()][0].values,list_products_13[data_all[TARGET].isnull()][1].values,list_products_13[data_all[TARGET].isnull()][2].values,list_products_13[data_all[TARGET].isnull()][3].values,
        list_products_14[data_all[TARGET].isnull()][0].values,list_products_14[data_all[TARGET].isnull()][1].values,list_products_14[data_all[TARGET].isnull()][2].values,list_products_14[data_all[TARGET].isnull()][3].values,
        list_products_15[data_all[TARGET].isnull()][0].values,list_products_15[data_all[TARGET].isnull()][1].values,list_products_15[data_all[TARGET].isnull()][2].values,list_products_15[data_all[TARGET].isnull()][3].values,
        list_products_16[data_all[TARGET].isnull()][0].values,list_products_16[data_all[TARGET].isnull()][1].values,list_products_16[data_all[TARGET].isnull()][2].values,list_products_16[data_all[TARGET].isnull()][3].values,
        list_products_17[data_all[TARGET].isnull()][0].values,list_products_17[data_all[TARGET].isnull()][1].values,list_products_17[data_all[TARGET].isnull()][2].values,list_products_17[data_all[TARGET].isnull()][3].values,
        list_products_18[data_all[TARGET].isnull()][0].values,list_products_18[data_all[TARGET].isnull()][1].values,list_products_18[data_all[TARGET].isnull()][2].values,list_products_18[data_all[TARGET].isnull()][3].values,
        list_products_19[data_all[TARGET].isnull()][0].values,list_products_19[data_all[TARGET].isnull()][1].values,list_products_19[data_all[TARGET].isnull()][2].values,list_products_19[data_all[TARGET].isnull()][3].values,
        list_products_20[data_all[TARGET].isnull()][0].values,list_products_20[data_all[TARGET].isnull()][1].values,list_products_20[data_all[TARGET].isnull()][2].values,list_products_20[data_all[TARGET].isnull()][3].values,
        list_products_21[data_all[TARGET].isnull()][0].values,list_products_21[data_all[TARGET].isnull()][1].values,list_products_21[data_all[TARGET].isnull()][2].values,list_products_21[data_all[TARGET].isnull()][3].values,
        list_products_22[data_all[TARGET].isnull()][0].values,list_products_22[data_all[TARGET].isnull()][1].values,list_products_22[data_all[TARGET].isnull()][2].values,list_products_22[data_all[TARGET].isnull()][3].values,
        list_products_23[data_all[TARGET].isnull()][0].values,list_products_23[data_all[TARGET].isnull()][1].values,list_products_23[data_all[TARGET].isnull()][2].values,list_products_23[data_all[TARGET].isnull()][3].values,
        list_products_24[data_all[TARGET].isnull()][0].values,list_products_24[data_all[TARGET].isnull()][1].values,list_products_24[data_all[TARGET].isnull()][2].values,list_products_24[data_all[TARGET].isnull()][3].values,
        list_products_25[data_all[TARGET].isnull()][0].values,list_products_25[data_all[TARGET].isnull()][1].values,list_products_25[data_all[TARGET].isnull()][2].values,list_products_25[data_all[TARGET].isnull()][3].values,
        list_products_26[data_all[TARGET].isnull()][0].values,list_products_26[data_all[TARGET].isnull()][1].values,list_products_26[data_all[TARGET].isnull()][2].values,list_products_26[data_all[TARGET].isnull()][3].values,
        list_products_27[data_all[TARGET].isnull()][0].values,list_products_27[data_all[TARGET].isnull()][1].values,list_products_27[data_all[TARGET].isnull()][2].values,list_products_27[data_all[TARGET].isnull()][3].values,
        list_products_28[data_all[TARGET].isnull()][0].values,list_products_28[data_all[TARGET].isnull()][1].values,list_products_28[data_all[TARGET].isnull()][2].values,list_products_28[data_all[TARGET].isnull()][3].values,
        list_products_29[data_all[TARGET].isnull()][0].values,list_products_29[data_all[TARGET].isnull()][1].values,list_products_29[data_all[TARGET].isnull()][2].values,list_products_29[data_all[TARGET].isnull()][3].values,
        list_products_30[data_all[TARGET].isnull()][0].values,list_products_30[data_all[TARGET].isnull()][1].values,list_products_30[data_all[TARGET].isnull()][2].values,list_products_30[data_all[TARGET].isnull()][3].values,
        list_products_31[data_all[TARGET].isnull()][0].values,list_products_31[data_all[TARGET].isnull()][1].values,list_products_31[data_all[TARGET].isnull()][2].values,list_products_31[data_all[TARGET].isnull()][3].values,
        list_products_32[data_all[TARGET].isnull()][0].values,list_products_32[data_all[TARGET].isnull()][1].values,list_products_32[data_all[TARGET].isnull()][2].values,list_products_32[data_all[TARGET].isnull()][3].values,
        list_products_33[data_all[TARGET].isnull()][0].values,list_products_33[data_all[TARGET].isnull()][1].values,list_products_33[data_all[TARGET].isnull()][2].values,list_products_33[data_all[TARGET].isnull()][3].values,
        list_products_34[data_all[TARGET].isnull()][0].values,list_products_34[data_all[TARGET].isnull()][1].values,list_products_34[data_all[TARGET].isnull()][2].values,list_products_34[data_all[TARGET].isnull()][3].values,
        list_products_35[data_all[TARGET].isnull()][0].values,list_products_35[data_all[TARGET].isnull()][1].values,list_products_35[data_all[TARGET].isnull()][2].values,list_products_35[data_all[TARGET].isnull()][3].values,
        list_products_36[data_all[TARGET].isnull()][0].values,list_products_36[data_all[TARGET].isnull()][1].values,list_products_36[data_all[TARGET].isnull()][2].values,list_products_36[data_all[TARGET].isnull()][3].values,
        list_products_37[data_all[TARGET].isnull()][0].values,list_products_37[data_all[TARGET].isnull()][1].values,list_products_37[data_all[TARGET].isnull()][2].values,list_products_37[data_all[TARGET].isnull()][3].values,
        list_products_38[data_all[TARGET].isnull()][0].values,list_products_38[data_all[TARGET].isnull()][1].values,list_products_38[data_all[TARGET].isnull()][2].values,list_products_38[data_all[TARGET].isnull()][3].values,
        list_products_39[data_all[TARGET].isnull()][0].values,list_products_39[data_all[TARGET].isnull()][1].values,list_products_39[data_all[TARGET].isnull()][2].values,list_products_39[data_all[TARGET].isnull()][3].values,
        list_products_40[data_all[TARGET].isnull()][0].values,list_products_40[data_all[TARGET].isnull()][1].values,list_products_40[data_all[TARGET].isnull()][2].values,list_products_40[data_all[TARGET].isnull()][3].values,
        list_products_41[data_all[TARGET].isnull()][0].values,list_products_41[data_all[TARGET].isnull()][1].values,list_products_41[data_all[TARGET].isnull()][2].values,list_products_41[data_all[TARGET].isnull()][3].values,
        list_products_42[data_all[TARGET].isnull()][0].values,list_products_42[data_all[TARGET].isnull()][1].values,list_products_42[data_all[TARGET].isnull()][2].values,list_products_42[data_all[TARGET].isnull()][3].values,
        list_products_43[data_all[TARGET].isnull()][0].values,list_products_43[data_all[TARGET].isnull()][1].values,list_products_43[data_all[TARGET].isnull()][2].values,list_products_43[data_all[TARGET].isnull()][3].values,
        data_all[data_all[TARGET].isnull()]['dif_time'].values, data_all[data_all[TARGET].isnull()]['num_products'].values
        ]

train_target = data_all[~data_all[TARGET].isnull()]['target_int'].values
train_index_np = data_all[~data_all[TARGET].isnull()][ID].values
test_index_np = data_all[data_all[TARGET].isnull()][ID].values



np_category = np.concatenate([np.zeros((1,4)), np.random.uniform(low=-0.05, high=0.05, size=(len(dict_category_int) - 1, 4))])
np_sub_category = np.concatenate([np.zeros((1,10)), np.random.uniform(low=-0.05, high=0.05, size=(len(dict_sub_category_int) - 1, 10))])
np_sub_sub_category = np.concatenate([np.zeros((1,15)), np.random.uniform(low=-0.05, high=0.05, size=(len(dict_sub_sub_category_int) - 1, 15))])
np_product_final = np.concatenate([np.zeros((1,20)), np.random.uniform(low=-0.05, high=0.05, size=(len(dict_product_final_int) - 1, 20))])




from keras.layers import Dropout, Flatten, concatenate, Subtract, Add, Activation
from keras.optimizers import Adam
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
from keras.layers import Input, Embedding, Dense
from keras.models import Model
from keras.callbacks import LearningRateScheduler, EarlyStopping, CSVLogger, ModelCheckpoint, ReduceLROnPlateau
from keras import backend as K


def keras_model():
    emb_category = Embedding(name='category_embedding', input_dim=np_category.shape[0], output_dim=np_category.shape[1], weights=[np_category], trainable=True, embeddings_regularizer=l2(1e-2))
    emb_sub_category = Embedding(name='sub_category', input_dim=np_sub_category.shape[0], output_dim=np_sub_category.shape[1], weights=[np_sub_category], trainable=True, embeddings_regularizer=l2(1e-2))
    emb_sub_sub_category = Embedding(name='sub_sub_category', input_dim=np_sub_sub_category.shape[0], output_dim=np_sub_sub_category.shape[1], weights=[np_sub_sub_category], trainable=True, embeddings_regularizer=l2(1e-2))
    emb_product_final = Embedding(name='product_final', input_dim=np_product_final.shape[0], output_dim=np_product_final.shape[1], weights=[np_product_final], trainable=True, embeddings_regularizer=l2(1e-2))

    ic1=Input(shape=[1]);isc1=Input(shape=[1]);issc1=Input(shape=[1]);ipf1=Input(shape=[1])
    ic2=Input(shape=[1]);isc2=Input(shape=[1]);issc2=Input(shape=[1]);ipf2=Input(shape=[1])
    ic3=Input(shape=[1]);isc3=Input(shape=[1]);issc3=Input(shape=[1]);ipf3=Input(shape=[1])
    ic4=Input(shape=[1]);isc4=Input(shape=[1]);issc4=Input(shape=[1]);ipf4=Input(shape=[1])
    ic5=Input(shape=[1]);isc5=Input(shape=[1]);issc5=Input(shape=[1]);ipf5=Input(shape=[1])
    ic6=Input(shape=[1]);isc6=Input(shape=[1]);issc6=Input(shape=[1]);ipf6=Input(shape=[1])
    ic7=Input(shape=[1]);isc7=Input(shape=[1]);issc7=Input(shape=[1]);ipf7=Input(shape=[1])
    ic8=Input(shape=[1]);isc8=Input(shape=[1]);issc8=Input(shape=[1]);ipf8=Input(shape=[1])
    ic9=Input(shape=[1]);isc9=Input(shape=[1]);issc9=Input(shape=[1]);ipf9=Input(shape=[1])
    ic10=Input(shape=[1]);isc10=Input(shape=[1]);issc10=Input(shape=[1]);ipf10=Input(shape=[1])
    ic11=Input(shape=[1]);isc11=Input(shape=[1]);issc11=Input(shape=[1]);ipf11=Input(shape=[1])
    ic12=Input(shape=[1]);isc12=Input(shape=[1]);issc12=Input(shape=[1]);ipf12=Input(shape=[1])
    ic13=Input(shape=[1]);isc13=Input(shape=[1]);issc13=Input(shape=[1]);ipf13=Input(shape=[1])
    ic14=Input(shape=[1]);isc14=Input(shape=[1]);issc14=Input(shape=[1]);ipf14=Input(shape=[1])
    ic15=Input(shape=[1]);isc15=Input(shape=[1]);issc15=Input(shape=[1]);ipf15=Input(shape=[1])
    ic16=Input(shape=[1]);isc16=Input(shape=[1]);issc16=Input(shape=[1]);ipf16=Input(shape=[1])
    ic17=Input(shape=[1]);isc17=Input(shape=[1]);issc17=Input(shape=[1]);ipf17=Input(shape=[1])
    ic18=Input(shape=[1]);isc18=Input(shape=[1]);issc18=Input(shape=[1]);ipf18=Input(shape=[1])
    ic19=Input(shape=[1]);isc19=Input(shape=[1]);issc19=Input(shape=[1]);ipf19=Input(shape=[1])
    ic20=Input(shape=[1]);isc20=Input(shape=[1]);issc20=Input(shape=[1]);ipf20=Input(shape=[1])
    ic21=Input(shape=[1]);isc21=Input(shape=[1]);issc21=Input(shape=[1]);ipf21=Input(shape=[1])
    ic22=Input(shape=[1]);isc22=Input(shape=[1]);issc22=Input(shape=[1]);ipf22=Input(shape=[1])
    ic23=Input(shape=[1]);isc23=Input(shape=[1]);issc23=Input(shape=[1]);ipf23=Input(shape=[1])
    ic24=Input(shape=[1]);isc24=Input(shape=[1]);issc24=Input(shape=[1]);ipf24=Input(shape=[1])
    ic25=Input(shape=[1]);isc25=Input(shape=[1]);issc25=Input(shape=[1]);ipf25=Input(shape=[1])
    ic26=Input(shape=[1]);isc26=Input(shape=[1]);issc26=Input(shape=[1]);ipf26=Input(shape=[1])
    ic27=Input(shape=[1]);isc27=Input(shape=[1]);issc27=Input(shape=[1]);ipf27=Input(shape=[1])
    ic28=Input(shape=[1]);isc28=Input(shape=[1]);issc28=Input(shape=[1]);ipf28=Input(shape=[1])
    ic29=Input(shape=[1]);isc29=Input(shape=[1]);issc29=Input(shape=[1]);ipf29=Input(shape=[1])
    ic30=Input(shape=[1]);isc30=Input(shape=[1]);issc30=Input(shape=[1]);ipf30=Input(shape=[1])
    ic31=Input(shape=[1]);isc31=Input(shape=[1]);issc31=Input(shape=[1]);ipf31=Input(shape=[1])
    ic32=Input(shape=[1]);isc32=Input(shape=[1]);issc32=Input(shape=[1]);ipf32=Input(shape=[1])
    ic33=Input(shape=[1]);isc33=Input(shape=[1]);issc33=Input(shape=[1]);ipf33=Input(shape=[1])
    ic34=Input(shape=[1]);isc34=Input(shape=[1]);issc34=Input(shape=[1]);ipf34=Input(shape=[1])
    ic35=Input(shape=[1]);isc35=Input(shape=[1]);issc35=Input(shape=[1]);ipf35=Input(shape=[1])
    ic36=Input(shape=[1]);isc36=Input(shape=[1]);issc36=Input(shape=[1]);ipf36=Input(shape=[1])
    ic37=Input(shape=[1]);isc37=Input(shape=[1]);issc37=Input(shape=[1]);ipf37=Input(shape=[1])
    ic38=Input(shape=[1]);isc38=Input(shape=[1]);issc38=Input(shape=[1]);ipf38=Input(shape=[1])
    ic39=Input(shape=[1]);isc39=Input(shape=[1]);issc39=Input(shape=[1]);ipf39=Input(shape=[1])
    ic40=Input(shape=[1]);isc40=Input(shape=[1]);issc40=Input(shape=[1]);ipf40=Input(shape=[1])
    ic41=Input(shape=[1]);isc41=Input(shape=[1]);issc41=Input(shape=[1]);ipf41=Input(shape=[1])
    ic42=Input(shape=[1]);isc42=Input(shape=[1]);issc42=Input(shape=[1]);ipf42=Input(shape=[1])
    ic43=Input(shape=[1]);isc43=Input(shape=[1]);issc43=Input(shape=[1]);ipf43=Input(shape=[1])
    
    
    ec1=emb_category(ic1);esc1=emb_sub_category(isc1);essc1=emb_sub_sub_category(issc1);epf1=emb_product_final(ipf1)
    ec2=emb_category(ic2);esc2=emb_sub_category(isc2);essc2=emb_sub_sub_category(issc2);epf2=emb_product_final(ipf2)
    ec3=emb_category(ic3);esc3=emb_sub_category(isc3);essc3=emb_sub_sub_category(issc3);epf3=emb_product_final(ipf3)
    ec4=emb_category(ic4);esc4=emb_sub_category(isc4);essc4=emb_sub_sub_category(issc4);epf4=emb_product_final(ipf4)
    ec5=emb_category(ic5);esc5=emb_sub_category(isc5);essc5=emb_sub_sub_category(issc5);epf5=emb_product_final(ipf5)
    ec6=emb_category(ic6);esc6=emb_sub_category(isc6);essc6=emb_sub_sub_category(issc6);epf6=emb_product_final(ipf6)
    ec7=emb_category(ic7);esc7=emb_sub_category(isc7);essc7=emb_sub_sub_category(issc7);epf7=emb_product_final(ipf7)
    ec8=emb_category(ic8);esc8=emb_sub_category(isc8);essc8=emb_sub_sub_category(issc8);epf8=emb_product_final(ipf8)
    ec9=emb_category(ic9);esc9=emb_sub_category(isc9);essc9=emb_sub_sub_category(issc9);epf9=emb_product_final(ipf9)
    ec10=emb_category(ic10);esc10=emb_sub_category(isc10);essc10=emb_sub_sub_category(issc10);epf10=emb_product_final(ipf10)
    ec11=emb_category(ic11);esc11=emb_sub_category(isc11);essc11=emb_sub_sub_category(issc11);epf11=emb_product_final(ipf11)
    ec12=emb_category(ic12);esc12=emb_sub_category(isc12);essc12=emb_sub_sub_category(issc12);epf12=emb_product_final(ipf12)
    ec13=emb_category(ic13);esc13=emb_sub_category(isc13);essc13=emb_sub_sub_category(issc13);epf13=emb_product_final(ipf13)
    ec14=emb_category(ic14);esc14=emb_sub_category(isc14);essc14=emb_sub_sub_category(issc14);epf14=emb_product_final(ipf14)
    ec15=emb_category(ic15);esc15=emb_sub_category(isc15);essc15=emb_sub_sub_category(issc15);epf15=emb_product_final(ipf15)
    ec16=emb_category(ic16);esc16=emb_sub_category(isc16);essc16=emb_sub_sub_category(issc16);epf16=emb_product_final(ipf16)
    ec17=emb_category(ic17);esc17=emb_sub_category(isc17);essc17=emb_sub_sub_category(issc17);epf17=emb_product_final(ipf17)
    ec18=emb_category(ic18);esc18=emb_sub_category(isc18);essc18=emb_sub_sub_category(issc18);epf18=emb_product_final(ipf18)
    ec19=emb_category(ic19);esc19=emb_sub_category(isc19);essc19=emb_sub_sub_category(issc19);epf19=emb_product_final(ipf19)
    ec20=emb_category(ic20);esc20=emb_sub_category(isc20);essc20=emb_sub_sub_category(issc20);epf20=emb_product_final(ipf20)
    ec21=emb_category(ic21);esc21=emb_sub_category(isc21);essc21=emb_sub_sub_category(issc21);epf21=emb_product_final(ipf21)
    ec22=emb_category(ic22);esc22=emb_sub_category(isc22);essc22=emb_sub_sub_category(issc22);epf22=emb_product_final(ipf22)
    ec23=emb_category(ic23);esc23=emb_sub_category(isc23);essc23=emb_sub_sub_category(issc23);epf23=emb_product_final(ipf23)
    ec24=emb_category(ic24);esc24=emb_sub_category(isc24);essc24=emb_sub_sub_category(issc24);epf24=emb_product_final(ipf24)
    ec25=emb_category(ic25);esc25=emb_sub_category(isc25);essc25=emb_sub_sub_category(issc25);epf25=emb_product_final(ipf25)
    ec26=emb_category(ic26);esc26=emb_sub_category(isc26);essc26=emb_sub_sub_category(issc26);epf26=emb_product_final(ipf26)
    ec27=emb_category(ic27);esc27=emb_sub_category(isc27);essc27=emb_sub_sub_category(issc27);epf27=emb_product_final(ipf27)
    ec28=emb_category(ic28);esc28=emb_sub_category(isc28);essc28=emb_sub_sub_category(issc28);epf28=emb_product_final(ipf28)
    ec29=emb_category(ic29);esc29=emb_sub_category(isc29);essc29=emb_sub_sub_category(issc29);epf29=emb_product_final(ipf29)
    ec30=emb_category(ic30);esc30=emb_sub_category(isc30);essc30=emb_sub_sub_category(issc30);epf30=emb_product_final(ipf30)
    ec31=emb_category(ic31);esc31=emb_sub_category(isc31);essc31=emb_sub_sub_category(issc31);epf31=emb_product_final(ipf31)
    ec32=emb_category(ic32);esc32=emb_sub_category(isc32);essc32=emb_sub_sub_category(issc32);epf32=emb_product_final(ipf32)
    ec33=emb_category(ic33);esc33=emb_sub_category(isc33);essc33=emb_sub_sub_category(issc33);epf33=emb_product_final(ipf33)
    ec34=emb_category(ic34);esc34=emb_sub_category(isc34);essc34=emb_sub_sub_category(issc34);epf34=emb_product_final(ipf34)
    ec35=emb_category(ic35);esc35=emb_sub_category(isc35);essc35=emb_sub_sub_category(issc35);epf35=emb_product_final(ipf35)
    ec36=emb_category(ic36);esc36=emb_sub_category(isc36);essc36=emb_sub_sub_category(issc36);epf36=emb_product_final(ipf36)
    ec37=emb_category(ic37);esc37=emb_sub_category(isc37);essc37=emb_sub_sub_category(issc37);epf37=emb_product_final(ipf37)
    ec38=emb_category(ic38);esc38=emb_sub_category(isc38);essc38=emb_sub_sub_category(issc38);epf38=emb_product_final(ipf38)
    ec39=emb_category(ic39);esc39=emb_sub_category(isc39);essc39=emb_sub_sub_category(issc39);epf39=emb_product_final(ipf39)
    ec40=emb_category(ic40);esc40=emb_sub_category(isc40);essc40=emb_sub_sub_category(issc40);epf40=emb_product_final(ipf40)
    ec41=emb_category(ic41);esc41=emb_sub_category(isc41);essc41=emb_sub_sub_category(issc41);epf41=emb_product_final(ipf41)
    ec42=emb_category(ic42);esc42=emb_sub_category(isc42);essc42=emb_sub_sub_category(issc42);epf42=emb_product_final(ipf42)
    ec43=emb_category(ic43);esc43=emb_sub_category(isc43);essc43=emb_sub_sub_category(issc43);epf43=emb_product_final(ipf43)
    
    vc1=Flatten()(ec1);vsc1=Flatten()(esc1);vssc1=Flatten()(essc1);vpf1=Flatten()(epf1)
    vc2=Flatten()(ec2);vsc2=Flatten()(esc2);vssc2=Flatten()(essc2);vpf2=Flatten()(epf2)
    vc3=Flatten()(ec3);vsc3=Flatten()(esc3);vssc3=Flatten()(essc3);vpf3=Flatten()(epf3)
    vc4=Flatten()(ec4);vsc4=Flatten()(esc4);vssc4=Flatten()(essc4);vpf4=Flatten()(epf4)
    vc5=Flatten()(ec5);vsc5=Flatten()(esc5);vssc5=Flatten()(essc5);vpf5=Flatten()(epf5)
    vc6=Flatten()(ec6);vsc6=Flatten()(esc6);vssc6=Flatten()(essc6);vpf6=Flatten()(epf6)
    vc7=Flatten()(ec7);vsc7=Flatten()(esc7);vssc7=Flatten()(essc7);vpf7=Flatten()(epf7)
    vc8=Flatten()(ec8);vsc8=Flatten()(esc8);vssc8=Flatten()(essc8);vpf8=Flatten()(epf8)
    vc9=Flatten()(ec9);vsc9=Flatten()(esc9);vssc9=Flatten()(essc9);vpf9=Flatten()(epf9)
    vc10=Flatten()(ec10);vsc10=Flatten()(esc10);vssc10=Flatten()(essc10);vpf10=Flatten()(epf10)
    vc11=Flatten()(ec11);vsc11=Flatten()(esc11);vssc11=Flatten()(essc11);vpf11=Flatten()(epf11)
    vc12=Flatten()(ec12);vsc12=Flatten()(esc12);vssc12=Flatten()(essc12);vpf12=Flatten()(epf12)
    vc13=Flatten()(ec13);vsc13=Flatten()(esc13);vssc13=Flatten()(essc13);vpf13=Flatten()(epf13)
    vc14=Flatten()(ec14);vsc14=Flatten()(esc14);vssc14=Flatten()(essc14);vpf14=Flatten()(epf14)
    vc15=Flatten()(ec15);vsc15=Flatten()(esc15);vssc15=Flatten()(essc15);vpf15=Flatten()(epf15)
    vc16=Flatten()(ec16);vsc16=Flatten()(esc16);vssc16=Flatten()(essc16);vpf16=Flatten()(epf16)
    vc17=Flatten()(ec17);vsc17=Flatten()(esc17);vssc17=Flatten()(essc17);vpf17=Flatten()(epf17)
    vc18=Flatten()(ec18);vsc18=Flatten()(esc18);vssc18=Flatten()(essc18);vpf18=Flatten()(epf18)
    vc19=Flatten()(ec19);vsc19=Flatten()(esc19);vssc19=Flatten()(essc19);vpf19=Flatten()(epf19)
    vc20=Flatten()(ec20);vsc20=Flatten()(esc20);vssc20=Flatten()(essc20);vpf20=Flatten()(epf20)
    vc21=Flatten()(ec21);vsc21=Flatten()(esc21);vssc21=Flatten()(essc21);vpf21=Flatten()(epf21)
    vc22=Flatten()(ec22);vsc22=Flatten()(esc22);vssc22=Flatten()(essc22);vpf22=Flatten()(epf22)
    vc23=Flatten()(ec23);vsc23=Flatten()(esc23);vssc23=Flatten()(essc23);vpf23=Flatten()(epf23)
    vc24=Flatten()(ec24);vsc24=Flatten()(esc24);vssc24=Flatten()(essc24);vpf24=Flatten()(epf24)
    vc25=Flatten()(ec25);vsc25=Flatten()(esc25);vssc25=Flatten()(essc25);vpf25=Flatten()(epf25)
    vc26=Flatten()(ec26);vsc26=Flatten()(esc26);vssc26=Flatten()(essc26);vpf26=Flatten()(epf26)
    vc27=Flatten()(ec27);vsc27=Flatten()(esc27);vssc27=Flatten()(essc27);vpf27=Flatten()(epf27)
    vc28=Flatten()(ec28);vsc28=Flatten()(esc28);vssc28=Flatten()(essc28);vpf28=Flatten()(epf28)
    vc29=Flatten()(ec29);vsc29=Flatten()(esc29);vssc29=Flatten()(essc29);vpf29=Flatten()(epf29)
    vc30=Flatten()(ec30);vsc30=Flatten()(esc30);vssc30=Flatten()(essc30);vpf30=Flatten()(epf30)
    vc31=Flatten()(ec31);vsc31=Flatten()(esc31);vssc31=Flatten()(essc31);vpf31=Flatten()(epf31)
    vc32=Flatten()(ec32);vsc32=Flatten()(esc32);vssc32=Flatten()(essc32);vpf32=Flatten()(epf32)
    vc33=Flatten()(ec33);vsc33=Flatten()(esc33);vssc33=Flatten()(essc33);vpf33=Flatten()(epf33)
    vc34=Flatten()(ec34);vsc34=Flatten()(esc34);vssc34=Flatten()(essc34);vpf34=Flatten()(epf34)
    vc35=Flatten()(ec35);vsc35=Flatten()(esc35);vssc35=Flatten()(essc35);vpf35=Flatten()(epf35)
    vc36=Flatten()(ec36);vsc36=Flatten()(esc36);vssc36=Flatten()(essc36);vpf36=Flatten()(epf36)
    vc37=Flatten()(ec37);vsc37=Flatten()(esc37);vssc37=Flatten()(essc37);vpf37=Flatten()(epf37)
    vc38=Flatten()(ec38);vsc38=Flatten()(esc38);vssc38=Flatten()(essc38);vpf38=Flatten()(epf38)
    vc39=Flatten()(ec39);vsc39=Flatten()(esc39);vssc39=Flatten()(essc39);vpf39=Flatten()(epf39)
    vc40=Flatten()(ec40);vsc40=Flatten()(esc40);vssc40=Flatten()(essc40);vpf40=Flatten()(epf40)
    vc41=Flatten()(ec41);vsc41=Flatten()(esc41);vssc41=Flatten()(essc41);vpf41=Flatten()(epf41)
    vc42=Flatten()(ec42);vsc42=Flatten()(esc42);vssc42=Flatten()(essc42);vpf42=Flatten()(epf42)
    vc43=Flatten()(ec43);vsc43=Flatten()(esc43);vssc43=Flatten()(essc43);vpf43=Flatten()(epf43)
    
    cp1=concatenate([vc1,vsc1,vssc1,vpf1])
    cp2=concatenate([vc2,vsc2,vssc2,vpf2])
    cp3=concatenate([vc3,vsc3,vssc3,vpf3])
    cp4=concatenate([vc4,vsc4,vssc4,vpf4])
    cp5=concatenate([vc5,vsc5,vssc5,vpf5])
    cp6=concatenate([vc6,vsc6,vssc6,vpf6])
    cp7=concatenate([vc7,vsc7,vssc7,vpf7])
    cp8=concatenate([vc8,vsc8,vssc8,vpf8])
    cp9=concatenate([vc9,vsc9,vssc9,vpf9])
    cp10=concatenate([vc10,vsc10,vssc10,vpf10])
    cp11=concatenate([vc11,vsc11,vssc11,vpf11])
    cp12=concatenate([vc12,vsc12,vssc12,vpf12])
    cp13=concatenate([vc13,vsc13,vssc13,vpf13])
    cp14=concatenate([vc14,vsc14,vssc14,vpf14])
    cp15=concatenate([vc15,vsc15,vssc15,vpf15])
    cp16=concatenate([vc16,vsc16,vssc16,vpf16])
    cp17=concatenate([vc17,vsc17,vssc17,vpf17])
    cp18=concatenate([vc18,vsc18,vssc18,vpf18])
    cp19=concatenate([vc19,vsc19,vssc19,vpf19])
    cp20=concatenate([vc20,vsc20,vssc20,vpf20])
    cp21=concatenate([vc21,vsc21,vssc21,vpf21])
    cp22=concatenate([vc22,vsc22,vssc22,vpf22])
    cp23=concatenate([vc23,vsc23,vssc23,vpf23])
    cp24=concatenate([vc24,vsc24,vssc24,vpf24])
    cp25=concatenate([vc25,vsc25,vssc25,vpf25])
    cp26=concatenate([vc26,vsc26,vssc26,vpf26])
    cp27=concatenate([vc27,vsc27,vssc27,vpf27])
    cp28=concatenate([vc28,vsc28,vssc28,vpf28])
    cp29=concatenate([vc29,vsc29,vssc29,vpf29])
    cp30=concatenate([vc30,vsc30,vssc30,vpf30])
    cp31=concatenate([vc31,vsc31,vssc31,vpf31])
    cp32=concatenate([vc32,vsc32,vssc32,vpf32])
    cp33=concatenate([vc33,vsc33,vssc33,vpf33])
    cp34=concatenate([vc34,vsc34,vssc34,vpf34])
    cp35=concatenate([vc35,vsc35,vssc35,vpf35])
    cp36=concatenate([vc36,vsc36,vssc36,vpf36])
    cp37=concatenate([vc37,vsc37,vssc37,vpf37])
    cp38=concatenate([vc38,vsc38,vssc38,vpf38])
    cp39=concatenate([vc39,vsc39,vssc39,vpf39])
    cp40=concatenate([vc40,vsc40,vssc40,vpf40])
    cp41=concatenate([vc41,vsc41,vssc41,vpf41])
    cp42=concatenate([vc42,vsc42,vssc42,vpf42])
    cp43=concatenate([vc43,vsc43,vssc43,vpf43])
    
    bcp1=BatchNormalization()(cp1)
    bcp2=BatchNormalization()(cp2)
    bcp3=BatchNormalization()(cp3)
    bcp4=BatchNormalization()(cp4)
    bcp5=BatchNormalization()(cp5)
    bcp6=BatchNormalization()(cp6)
    bcp7=BatchNormalization()(cp7)
    bcp8=BatchNormalization()(cp8)
    bcp9=BatchNormalization()(cp9)
    bcp10=BatchNormalization()(cp10)
    bcp11=BatchNormalization()(cp11)
    bcp12=BatchNormalization()(cp12)
    bcp13=BatchNormalization()(cp13)
    bcp14=BatchNormalization()(cp14)
    bcp15=BatchNormalization()(cp15)
    bcp16=BatchNormalization()(cp16)
    bcp17=BatchNormalization()(cp17)
    bcp18=BatchNormalization()(cp18)
    bcp19=BatchNormalization()(cp19)
    bcp20=BatchNormalization()(cp20)
    bcp21=BatchNormalization()(cp21)
    bcp22=BatchNormalization()(cp22)
    bcp23=BatchNormalization()(cp23)
    bcp24=BatchNormalization()(cp24)
    bcp25=BatchNormalization()(cp25)
    bcp26=BatchNormalization()(cp26)
    bcp27=BatchNormalization()(cp27)
    bcp28=BatchNormalization()(cp28)
    bcp29=BatchNormalization()(cp29)
    bcp30=BatchNormalization()(cp30)
    bcp31=BatchNormalization()(cp31)
    bcp32=BatchNormalization()(cp32)
    bcp33=BatchNormalization()(cp33)
    bcp34=BatchNormalization()(cp34)
    bcp35=BatchNormalization()(cp35)
    bcp36=BatchNormalization()(cp36)
    bcp37=BatchNormalization()(cp37)
    bcp38=BatchNormalization()(cp38)
    bcp39=BatchNormalization()(cp39)
    bcp40=BatchNormalization()(cp40)
    bcp41=BatchNormalization()(cp41)
    bcp42=BatchNormalization()(cp42)
    bcp43=BatchNormalization()(cp43)
    
    sum_products = Add()([bcp1,bcp2,bcp3,bcp4,bcp5,bcp6,bcp7,bcp8,bcp9,bcp10
                      ,bcp11,bcp12,bcp13,bcp14,bcp15,bcp16,bcp17,bcp18,bcp19
                      ,bcp20,bcp21,bcp22,bcp23,bcp24,bcp25,bcp26,bcp27,bcp28,
                      bcp29,bcp30,bcp31,bcp32,bcp33,bcp34,bcp35,bcp36,bcp37
                      ,bcp38,bcp39,bcp40,bcp41,bcp42,bcp43])
    
    item_vecs = Dropout(0.5)(sum_products)
    item_vecs = Dense(32, activation='relu')(item_vecs)

    i_time=Input(shape=[1]);ilen=Input(shape=[1])
    
    item_vecs = concatenate([item_vecs, i_time, ilen])
    item_vecs = BatchNormalization()(item_vecs)
    item_vecs = Dropout(0.5)(item_vecs)
    item_vecs = Dense(8, activation='relu')(item_vecs)
    item_vecs = BatchNormalization()(item_vecs)

    output = Dropout(0.5)(item_vecs)
    output = Dense(1, activation='sigmoid')(output)
    
    model = Model(inputs=[
            ic1,isc1,issc1,ipf1,
            ic2,isc2,issc2,ipf2,
            ic3,isc3,issc3,ipf3,
            ic4,isc4,issc4,ipf4,
            ic5,isc5,issc5,ipf5,
            ic6,isc6,issc6,ipf6,
            ic7,isc7,issc7,ipf7,
            ic8,isc8,issc8,ipf8,
            ic9,isc9,issc9,ipf9,
            ic10,isc10,issc10,ipf10,
            ic11,isc11,issc11,ipf11,
            ic12,isc12,issc12,ipf12,
            ic13,isc13,issc13,ipf13,
            ic14,isc14,issc14,ipf14,
            ic15,isc15,issc15,ipf15,
            ic16,isc16,issc16,ipf16,
            ic17,isc17,issc17,ipf17,
            ic18,isc18,issc18,ipf18,
            ic19,isc19,issc19,ipf19,
            ic20,isc20,issc20,ipf20,
            ic21,isc21,issc21,ipf21,
            ic22,isc22,issc22,ipf22,
            ic23,isc23,issc23,ipf23,
            ic24,isc24,issc24,ipf24,
            ic25,isc25,issc25,ipf25,
            ic26,isc26,issc26,ipf26,
            ic27,isc27,issc27,ipf27,
            ic28,isc28,issc28,ipf28,
            ic29,isc29,issc29,ipf29,
            ic30,isc30,issc30,ipf30,
            ic31,isc31,issc31,ipf31,
            ic32,isc32,issc32,ipf32,
            ic33,isc33,issc33,ipf33,
            ic34,isc34,issc34,ipf34,
            ic35,isc35,issc35,ipf35,
            ic36,isc36,issc36,ipf36,
            ic37,isc37,issc37,ipf37,
            ic38,isc38,issc38,ipf38,
            ic39,isc39,issc39,ipf39,
            ic40,isc40,issc40,ipf40,
            ic41,isc41,issc41,ipf41,
            ic42,isc42,issc42,ipf42,
            ic43,isc43,issc43,ipf43,
            i_time, ilen], outputs=output)
    
    model.compile(optimizer=Adam(lr=1e-2, decay=1e-3), loss='binary_crossentropy',
                  metrics=['accuracy'])
    
    return model




from sklearn.model_selection import StratifiedKFold

cv_list = [StratifiedKFold(n_splits=NUM_CV, shuffle=True, random_state=x+123) for x in range(NUM_CV_LIST)]



cv_index = -1
for cv_ in cv_list:
    cv_index += 1
    cv___ = -1
    for train_index, val_index in cv_.split(train_target, train_target):
        cv___ += 1
        
        print(cv_index)
        print(cv___)
        
        tr_data_list = [x[train_index] for x in data_train_list]
        val_data_list = [x[val_index] for x in data_train_list]
        
        model = keras_model()
        
        log_model = CSVLogger(PATH_MODEL + 'eval_model_{}_{}.log'.format(cv_index, cv___), append=True)
        checkpointer = ModelCheckpoint(filepath=PATH_MODEL + 'model_{}_{}.hdf5'.format(cv_index, cv___), verbose=0, save_best_only=True)
        earlyStopping = EarlyStopping(monitor='val_loss',  patience=100, verbose=0, mode='auto', min_delta=1e-4)
        learning_rate_reduction = ReduceLROnPlateau(monitor='val_loss', patience=20, verbose=0, factor=0.5, min_lr=1e-30)

        
        model.fit(tr_data_list, train_target[train_index],
                  verbose=0,
                  epochs=EPOCH,
                  validation_data=(val_data_list, train_target[val_index]),
                  batch_size=BATCH_SIZE,
                  callbacks=[log_model, checkpointer, earlyStopping,
                             learning_rate_reduction],
                             shuffle=True)
        
        best_model = keras_model()
        best_model.load_weights(PATH_MODEL + 'model_{}_{}.hdf5'.format(cv_index, cv___))
        
        pred_val = best_model.predict(val_data_list)
        pred_test = best_model.predict(data_test_list)
        
        pred_val_df = pd.DataFrame(pred_val, index=train_index_np[val_index])
        pred_val_df[TARGET] = ['male' if x == 1 else 'female' for x in train_target[val_index]]
        pred_val_df.to_csv(PATH_MODEL + 'cv_{}_{}.csv'.format(cv_index, cv___))
        
        pred_test_df = pd.DataFrame(pred_test, index=test_index_np)
        pred_test_df.to_csv(PATH_MODEL + 'test_{}_{}.csv'.format(cv_index, cv___))
        
        del model
        del best_model
        K.clear_session()

        



