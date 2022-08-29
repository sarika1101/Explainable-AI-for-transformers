import pandas as pd
import numpy as np
import json

import nltk
from nltk.corpus import stopwords
from nltk  import WordNetLemmatizer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk import pos_tag, word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer

#import seaborn for plotting
import seaborn as sns

#import libraries for model evaluation
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import accuracy_score, f1_score
from sklearn.linear_model import LogisticRegression
from simpletransformers.classification import ClassificationModel

import torch
import os, sys, stat
import util
from util import util

import time
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score

def eda(data):
    ob = util()
    print("We will start the EDA")
    print(" The dataset contains ",data.shape[0]," total rows and ",data.shape[1]," total number of columns")
    
    print("Total categories in this dataset are ")
    print(data['cyberbullying_type'].value_counts())
    
    print(" Lets see number of duplicate records present in the dataset")
    df_dup = data[data.duplicated(subset = ['tweet_text'])]
    print(df_dup['cyberbullying_type'].value_counts())
    
    print(" We will drop all duplicates")
    data.drop_duplicates(subset=['tweet_text'],inplace = True)
    
    print("We will check the null values")
    print(data.isna().sum())
    
    print("There are no null values present")
    
    print("We will clean the data for model training")
    
    data_clean = ob.data_clean(data)
    print("tweet text is cleaned")
    return data_clean

 #Converting the target variable into integer
def target_vector(ob,data,output_folder):
    label_name_list = data['cyberbullying_type'].unique()
    
    label_dict = {}
    label_dict_rev = {}
    label_count = 0
    
    for label_name in label_name_list:
        label_dict[label_name] = label_count
        label_dict_rev[label_count] = label_name
        label_count+=1
    
    data['cyber_type'] = data['cyberbullying_type'].map(label_dict)
    
    for key,val in label_dict.items():
        print(key,":",val)
    
    ob.dump_json(label_dict_rev,output_folder)
    
    label_list = data['cyber_type'].tolist()
    
    return data,label_list,label_dict

def model_eval(ground_truth,output_pred,labels,target_names):
    
    cr_non_null = classification_report(ground_truth,output_pred,labels = labels, target_names = target_names)
    
    
    model_acc = accuracy_score(ground_truth,output_pred)
    
    fd = os.open("analysis.txt",os.O_RDWR| os.O_CREAT)
    analysis_file = os.fdopen(fd,"w+")
    
    analysis_file.write("Accuracy Score for the model:   ")
    analysis_file.write(str(cr_non_null))
    
    analysis_file.write("Classification Report :\n")
    analysis_file.write(str(model_acc))
    
    analysis_file.close()
    
    return


def model_train(model_name,output_folder,model_location,data,label_list,label_dict):
    #train test split
    df_train, df_test = train_test_split(data,test_size = 0.1, stratify = label_list)
    
    ground_truth = df_test['cyber_type'].tolist()
    test_text_topredict = df_test['fully_cleaned_tweet'].tolist()
    
    df_train.drop(columns = ['tweet_text','cyberbullying_type','cleaned_tweet'],inplace = True)
    
    model = ClassificationModel(model_name,
                           model_location,
                           num_labels = len(label_dict),
                           args = {"fp16":False, "num_train_epochs":4,"output_dir":output_folder, "train_batch_size":8,
                                  "eval_batch_size":8, "use_multiprocessing":False,"reprocess_input_data": True,
                                  "overwrite_output_dir":True, "save_steps": -1, "save_model_every_epoch":False,
                                  "save_eval_checkpoints":False},use_cuda = torch.cuda.is_available()
                           )
    
    st = time.time()
    print("Training model started in time : ",st," for model ",model_name)
    model.train_model(df_train)
    et = time.time()
    print("Training model ended in time : ",(et-st))
    
    ground_truth = df_test['cyber_type'].tolist()
    test_text_topredict = df_test['fully_cleaned_tweet'].tolist()
    
    target_names = list(label_dict.keys())
    labels = list(label_dict.values())
    
    output_pred ,raw_output = model.predict(test_text_topredict)
    
    model_eval(ground_truth,output_pred,labels,target_names)
    
    return model

if __name__ == "__main__":
    
    ob = util()
    data = ob.read_file("cyberbullying_tweets.csv")
    
    data = eda(data)
    print("Total number of records after cleaning : ",data.shape)
    print("Now we will start model training")
    #load the config file and print for test
    
    config_filename = r"config.json"
    config = ob.read_json(config_filename)
    
    
    #get model name and location of pre-trained from config file
    model_name = config['model']['model_name']
    print("Model name is : ",model_name)
    model_location = config['model']['model_location']
    output_folder = config['model']['output_folder']
    
    data,label_list,label_dict = target_vector(ob,data,output_folder)
    
    model = model_train(model_name,output_folder,model_location,data,label_list,label_dict)
    
    
    
    
    
    