#!/usr/bin/env python
# coding: utf-8

# In[32]:


import glob
import pandas as pd
# from tqdm import tqdm
import numpy as np
import os
import pickle
import sys
from pandas.api.types import is_numeric_dtype
from collections import Counter,defaultdict
import time
from sklearn.feature_extraction import DictVectorizer
import enchant
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.feature_extraction.text import CountVectorizer
from tabulate import tabulate 
from keras.preprocessing import text as keras_text, sequence as keras_seq
from keras.models import load_model

# In[33]:


#read csv
# dict_label = {'Usable directly numeric':0, 'Usable with extraction':1, 'Usable with Extration': 1, 'Usable with extraction ':1, 'Usable directly categorical':2, 'Unusable':3, 'Context_specific':4, 'Usable directly categorical ':2}
# data = pd.read_csv('data_for_ML_num.csv')

# data['y_act'] = [dict_label[i] for i in data['y_act']]
# y = data.loc[:,['y_act']]


# In[34]:


# data['Num of nans'] = [data['Num of nans'][i]*100/data['Total_val'][i] for i in data.index]
# data['num of dist_val'] = [data['num of dist_val'][i]*100/data['Total_val'][i] for i in data.index]

# data1 = data[['Num of nans', 'max_val', 'mean', 'min_val', 'num of dist_val','std_dev','castability','extractability', 'len_val']]
# data1 = data1.fillna(0)

# arr = data['Attribute_name'].values
# vectorizer = CountVectorizer(ngram_range=(3,3),analyzer='char')
# X = vectorizer.fit_transform(arr)
# pickle.dump(vectorizer, open("vector.pickel", "wb"))


# In[35]:


def summary_stats(dat, key_s):
    b_data = []
    for col in key_s:
        nans = np.count_nonzero(pd.isnull(dat[col]))
        dist_val = len(pd.unique(dat[col].dropna()))
        Total_val = len(dat[col])
        #print(Total_val)
        mean = 0
        std_dev = 0
        var = 0
        min_val = 0
        max_val = 0
        if is_numeric_dtype(dat[col]):
            mean = np.mean(dat[col])
            
            if pd.isnull(mean):
                mean = 0
                std_dev = 0
                #var = 0
                min_val = 0
                max_val = 0
                
            else:    
                std_dev = np.std(dat[col])
                var = np.var(dat[col])
                min_val = float(np.min(dat[col]))
                max_val = float(np.max(dat[col]))
        b_data.append([Total_val, nans, dist_val, mean, std_dev, min_val, max_val])
    return b_data

def castability_feature(dat, column_names):
    castability_list = []
    #make sure the value you are avaluating is not nan
    for keys in column_names:
        #print(keys)
        i = 0
        while pd.isnull(dat[keys][i]):
            i += 1
            if i > len(dat[keys]) - 2:
                break
        #if type is string try casting
        if dat[keys][i].__class__.__name__ == 'str':
            try:
                castability = str(type(eval(dat[keys][i])))
                castability_list.append(1)
            except:
                castability_list.append(0)
        else:
            castability_list.append(0)
    return castability_list  

def get_class_type(dat, column_names):
    as_read = []
    master_key_dictionary =  master_key()
    for keys in column_names:
        
        #make sure the value you are avaluating is not nan
        i = 0
        while pd.isnull(dat[keys][i]):
            i += 1
            if i > len(dat[keys]) - 2:
                break
        val = -1
        type_pyth = dat[keys][i].__class__.__name__
        for tipe in master_key_dictionary:
            if tipe in type_pyth:
                val = master_key_dictionary[tipe]
        as_read.append(val)
    return as_read

def master_key():
    master_key_dic = defaultdict(int)
    master_key_dic['str'] = 0
    master_key_dic['float'] = 1
    master_key_dic['int'] = 1
    return master_key_dic

def numeric_extraction(dat,column_names):
    #0 no , 1 yes
    numeric_extraction_list = []
    #make sure the value you are avaluating is not nan
    for keys in column_names:
        i = 0
        while pd.isnull(dat[keys][i]):
            i += 1
            if i > len(dat[keys]) - 2:
                break
        val = 0
            
        if dat[keys][i].__class__.__name__ == 'str':
            #print('yes')
            #check whether any number can be extracted
            try:
                #it will faile when you have no numbers or if you have two numbers seperated by space
                float(re.sub('[^0-9\. ]', ' ',dat[keys][i]))
                #print('yes')
                val = 1
            except:
                pass
            
        numeric_extraction_list.append(val)
    
    return numeric_extraction_list


def val_length(dat,column_names):
    val = []
    for keys in column_names:
        i = 0
        while pd.isnull(dat[keys][i]):
            i += 1
            if i > len(dat[keys]) - 2:
                break
        try:
            val.append(len(str(dat[keys][i]).split()))
        except UnicodeEncodeError:    
            val.append(len(str(dat[keys][i].encode('utf-8')).split()))
    return val      


# In[3]:

def get_sample(dat, key_s):
    rand = []
    for name in keys:
        rand_sample = list(pd.unique(dat[name]))
        rand_sample = rand_sample[:5]
        while len(rand_sample) < 5:
            rand_sample.append(list(pd.unique(dat[name]))[np.random.randint(len(list(pd.unique(dat[name]))))])
        rand.append(rand_sample[:5])
    return rand
     

def get_castability(dat):
    sample_values = dat['sample_1':'sample_5']
    #make sure the value you are avaluating is not nan
    castability = 0
    for values in sample_values:
        print(values.__class__.__name__)
        if pd.isnull(values) == False and (values.__class__.__name__ == 'str' or values.__class__.__name__ == 'unicode'):
            try:
                castability = str(type(eval(values)))
                castability = 1
                break
            except:
                dew = 1
                #print('passing')
#                 pass
        if pd.isnull(values) == False and (values.__class__.__name__ == 'int' or values.__class__.__name__ == 'float'):
            castability = 1
            break
            
    return castability  

def get_extractability(dat, cast):
    sample_values = dat['sample_1':'sample_5']
    #make sure the value you are avaluating is not nan
    extractability = 0
    for values in sample_values:
        if pd.isnull(values) == False and values.__class__.__name__ == 'str' and cast == 0:
            try:
                #it will faile when you have no numbers or if you have two numbers seperated by space
                float(re.sub('[^0-9\. ]', ' ',values))
                #print('yes')
                extractability = 1
                break
            except:
                pass
    return extractability 

def get_len(dat):
    sample_values = dat['sample_1':'sample_5']
    #make sure the value you are avaluating is not nan
    length = 0
    for values in sample_values:
        if pd.isnull(values) == False:
            length = len(str(values).split()) + length
    return length/5.0 

def bag_of_words_extraction(name_keys):
    vectoriser = DictVectorizer()
    word_vec = attr_bagofwords(name_keys)
    X = vectoriser.fit_transform(Counter(lst) for lst in word_vec)
    b_of_words = X.A
    dict_of_words = vectoriser.vocabulary_
    return b_of_words, dict_of_words
    
    
def attr_bagofwords(name_ke):
    b_words = []
    d = enchant.Dict("en_US")
    for name in name_ke:
            name = name.split()
            temp = []
            for i in name:
                if d.check(i.lower()):
                    temp.extend(i.lower())
                else:
                    temp.extend(d.suggest(i.lower()))
            b_words.append(temp)
    return b_words


# In[ ]:





# In[71]:


# Total_val, nans, dist_val, mean, std_dev, min_val, max_val
csv_names = ['Attribute_name','Total_val', 'Num of nans', 'num of dist_val', 'mean', 'std_dev', 'min_val', 'max_val','castability','extractability', 'len_val','sample_1', 'sample_2', 'sample_3','sample_4','sample_5']
golden_data = pd.DataFrame(columns = csv_names)
final_data = pd.DataFrame(columns = csv_names)
keys = []

def BaseFeaturization(CsvFile):
    
    path_name = CsvFile
    stats = []
    attribute_name = []
    sample = []
    csv_names = []
    id_value = []
    i = 0
    url_list = []
    castability = []
    class_type = []
    number_extraction = []
    value_length = []

    df = pd.read_csv(path_name,encoding = 'latin1',lineterminator='\n')
#     df= df[['BUILT_Population_2014']]
#     print(df)
    global keys
    keys = list(df.keys())

    attribute_name.extend(keys)
    print(attribute_name)
    stats.extend(summary_stats(df, keys))
    print(stats)
    sample.extend(get_sample(df,keys))
    print(sample)
    castability.extend(castability_feature(df, keys))
    print(castability)
    number_extraction.extend(numeric_extraction(df, keys))
    print(number_extraction)
    value_length.extend(val_length(df, keys))
    print(value_length)
    
    global golden_data, final_data
    print('---')
    val_append = []
    for i in range(len(stats)):
#         val_append = [id_value[i]] ## Record
#         val_append.append(0)  ## Label
        val_append = []
        val_append.append(attribute_name[i])
        val_append.extend(stats[i])
        val_append.append(castability[i])
        val_append.append(number_extraction[i])
        val_append.append(value_length[i])
        val_append.extend(sample[i])        
        print(val_append)
#         print(golden_data)
        golden_data.loc[i] = val_append
#         golden_data.loc[len(golden_data),:] = val_append
        # golden_data  = pd.concat([golden_data,val_append],axis=0)    
     
    castability, extractability, len_val = [], [], []
    for i in range(len(golden_data)):
        castability.append(get_castability(golden_data.loc[i]))
        # extractability.append(get_extractability(golden_data.loc[i], castability[i]))
        # len_val.append(get_len(golden_data.loc[i]))
    golden_data['castability'] = castability
    # golden_data['extractability'] = extractability
    # golden_data['len_val'] = len_val          

    golden_data = golden_data.rename(columns={'Num of nans': 'Num_of_nans', 'num of dist_val': 'num_of_dist_val'})
#     golden_data['Num_of_nans'] = [float(golden_data['Num_of_nans'][i])/float(golden_data['Total_val'][i]) for i in golden_data.index]
#     golden_data['num_of_dist_val'] = [float(golden_data['num_of_dist_val'][i])/float(golden_data['Total_val'][i]) for i in golden_data.index]
    
    golden_data['Num_of_nans'] = [golden_data['Num_of_nans'][i]*100/golden_data['Total_val'][i] for i in golden_data.index]
    golden_data['num_of_dist_val'] = [golden_data['num_of_dist_val'][i]*100/golden_data['Total_val'][i] for i in golden_data.index]    
    

    if rf:
        data1 = golden_data[['Num_of_nans', 'max_val', 'mean', 'min_val', 'num_of_dist_val','std_dev','castability','extractability', 'len_val']]
        data1 = data1.fillna(0)
    #     data1 = data1.rename(columns={'mean': 'scaled_mean', 'min_val': 'scaled_min_val', 'max_val': 'scaled_max_val','std_dev': 'scaled_std_dev'})

    #     column_names_to_normalize = ['scaled_max_val', 'scaled_mean', 'scaled_min_val','scaled_std_dev']
    #     # column_names_to_normalize = ['scaled_mean','scaled_std_dev', 'scaled_len_val']
    #     x = data1[column_names_to_normalize].values
    #     x = np.nan_to_num(x)
    #     x_scaled = StandardScaler().fit_transform(x)    
        
        
    #     df_temp = pd.DataFrame(x_scaled, columns=column_names_to_normalize, index = data1.index)
    #     data1[column_names_to_normalize] = df_temp
        data1.Num_of_nans = data1.Num_of_nans.astype(float)
        data1.num_of_dist_val = data1.num_of_dist_val.astype(float)
        data1.castability = data1.castability.astype(float)
        data1.extractability = data1.extractability.astype(float)
        
        d = enchant.Dict("en_US")
        for i in golden_data.index:
            ival = golden_data.at[i,'Attribute_name']
            if ival != 'id' and d.check(ival):
                data1.at[i,'dictionary_item'] = 1
            else:
                data1.at[i,'dictionary_item'] = 0

        final_data = data1
    
    if cnn:
#         data1 = data1.rename(columns={'mean': 'scaled_mean', 'min_val': 'scaled_min_val', 'max_val': 'scaled_max_val','std_dev': 'scaled_std_dev'})
#         column_names_to_normalize = ['scaled_max_val', 'scaled_mean', 'scaled_min_val','scaled_std_dev']
        column_names_to_normalize = ['max_val', 'mean', 'min_val','std_dev','num_of_dist_val','Num_of_nans','Total_val']        
        x = golden_data[column_names_to_normalize].values
        x = np.nan_to_num(x)
        x_scaled = StandardScaler().fit_transform(x)
        df_temp = pd.DataFrame(x_scaled, columns=column_names_to_normalize, index = golden_data.index)
        golden_data[column_names_to_normalize] = df_temp

        data1 = golden_data[['Num_of_nans', 'max_val', 'mean', 'min_val', 'num_of_dist_val','std_dev','castability','extractability', 'len_val','Total_val']]
        arr = golden_data['Attribute_name'].values
#         print(arr)
        arr1 = golden_data['sample_1'].values
        print(arr1)
        arr1_1 = []
        for x in arr1:
            try:
                arr1_1.append(str(x))
            except UnicodeEncodeError:
                arr1_1.append(str(x.encode('utf-8')))
        arr1 = arr1_1
        
        arr2 = golden_data['sample_2'].values
        print(arr2)
        arr2_1 = []
        for x in arr2:
            try:
                arr2_1.append(str(x))
            except UnicodeEncodeError:
                arr2_1.append(str(x.encode('utf-8')))
        arr2 = arr2_1        
        
#         arr1 = [str(x) for x in arr1]
#         try:
#             arr1 = [str(x) for x in arr1]
#         except UnicodeEncodeError:
#             arr1 = [str(x.encode('utf-8')) for x in arr1]
#         print(arr1)
#         arr2 = golden_data['sample_2'].values
#         arr2 = [str(x) for x in arr2]
#         print(arr2)
#         vectorizer = CountVectorizer(ngram_range=(2,2),analyzer='char')
        
        vectorizer1 = pickle.load(open("vectorcnn1.pickel", "rb"), encoding="latin1")
        X = vectorizer1.transform(arr)
#         print(len(vectorizer1.get_feature_names()))        
        
        vectorizer2 = pickle.load(open("vectorcnn2.pickel", "rb"), encoding="latin1")
        X1 = vectorizer2.transform(arr1)
#         print(len(vectorizer2.get_feature_names()))        
        
        vectorizer3 = pickle.load(open("vectorcnn3.pickel", "rb"), encoding="latin1")
        X2 = vectorizer3.transform(arr2)
#         print(len(vectorizer3.get_feature_names()))        

        # print(X.toarray())

        # data1.to_csv('before.csv')
#         print(vectorizer.get_feature_names())
        tempdf = pd.DataFrame(X.toarray())
        tempdf1 = pd.DataFrame(X1.toarray())
        tempdf2 = pd.DataFrame(X2.toarray())
        data2 = pd.concat([data1,tempdf,tempdf1,tempdf2], axis=1, sort=False)   
        data2 = data2.rename(columns={'mean': 'scaled_mean', 'min_val': 'scaled_min_val', 'max_val': 'scaled_max_val','std_dev': 'scaled_std_dev'})
            
        print(data1.shape)
        print(data2.shape)            

        final_data = data2


    return final_data


# In[72]:


def Featurize(data1, signal1,signal2='',signal3='',n=3):
#     vectorizer = CountVectorizer(ngram_range=(n,n),analyzer='char')
    vectorizer = pickle.load(open("vector.pickel", "rb"))
    data2 = pd.DataFrame() 
    
    global golden_data
    
    if signal1:
        arr1 = golden_data[signal1].values
        print(arr1)
        X1 = vectorizer.transform(arr1)
        print(X1)
        tempdf1 = pd.DataFrame(X1.toarray())
        data2 = pd.concat([data1,tempdf1], axis=1, sort=False)        
    if signal2:
        arr2 = golden_data[signal2].values
        arr2 = [str(x) for x in arr2]
        X2 = vectorizer.fit_transform(arr2)
        tempdf2 = pd.DataFrame(X2.toarray())
        data2 = pd.concat([data2,tempdf2], axis=1, sort=False)
    if signal3:
        arr3 = golden_data[signal3].values
        arr3 = [str(x) for x in arr3]
        X3 = vectorizer.fit_transform(arr3)
        tempdf3 = pd.DataFrame(X3.toarray())
        data2 = pd.concat([data2,tempdf3], axis=1, sort=False)
    
    data2.dropna(inplace=True)
    return data2


# In[73]:


# data1 = BaseFeaturization('insurance.csv')
# data2 = Featurize(data1, 'Attribute_name','','',3)


# # In[74]:


# data2.dropna(inplace=True)
# print(data2)


# In[75]:

str2return = ''
cnn,rf,knn = 0,0,0
def Initialize(curstr):
    global str2return,cnn,rf,knn
    if curstr == 'rf':
        str2return = 'rfmodel.sav'
        rf = 1
    elif curstr == 'neural':
        str2return = 'neuralmodel.h5'
        cnn = 1
    elif curstr == 'knn':
        str2return = 'knnmodel.sav'
        knn = 1

def LoadModel(data2):
    global final_data
    filename = str2return

    if rf:
        loaded_model = pickle.load(open(filename, 'rb'),encoding='latin1')
        y_prob = loaded_model.predict_proba(data2)
        print(y_prob)

        predictions = np.argmax(y_prob, axis=1)
        confidences = np.max(y_prob, axis=1)
        print(predictions)
        print(keys)
    
    if cnn:
        list_sentences_train = keys
        tokenizer = keras_text.Tokenizer(char_level = True)
        tokenizer.fit_on_texts(list(list_sentences_train))
        list_tokenized_train = tokenizer.texts_to_sequences(list_sentences_train)
        X_t = keras_seq.pad_sequences(list_tokenized_train, maxlen=512)   

        final_data = final_data.values
        bestone = load_model('neuralmodel.h5')
        y_prob = bestone.predict([X_t,final_data])
        print(y_prob)

        y_pred = [np.argmax(i) for i in y_prob]
        print(y_pred)

        predictions = np.argmax(y_prob, axis=1)
        confidences = np.max(y_prob, axis=1)
        print(predictions)
        print(keys)

    dict_label_inv = {0:'Numeric', 1:'Needs-Extraction', 2:'Categorical', 3:'Not-Generalizable', 4:'Context-Specific'}
    matrix = []
    i=0
    for x in predictions:
        templst = []
        templst.append(keys[i])
        templst.append(dict_label_inv[x])
        templst.append(confidences[i])
        matrix.append(templst)
        i=i+1 

    headers = ['Column', 'Inferred Feature Type', 'Confidence Score']
    print(tabulate(matrix,headers))

# In[ ]:




