import pandas as pd
import numpy as np
import json

import os, sys, stat

import nltk
from nltk.corpus import stopwords
from nltk  import WordNetLemmatizer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk import pos_tag, word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer

words_to_remove = ["ax","i","you","edu","s","t","m","subject","can","lines","re","what", "there","all","we",
                "one","the","a","an","of","or","in","for","by","on","but","is","in","a","not","with","as",
                "was","if","they","are","this","and","it","have","has","from","at","my","be","by","not","that","to",
                "from","com","org","like","likes","so","said","from","what","told","over","more","other",
                "have","last","with","this","that","such","when","been","says","will","also","where","why",
                "would","today", "in", "on", "you", "r", "d", "u", "hw","wat", "oly", "s", "b", "ht", 
                "rt", "p","the","th", "n", "was"]
class util:
    def write_file(self,location, data):
        print("Writing the file in the output folder")
        fd = os.open(location, os.O_RDWR | os.O_CREAT)
        fo = os.fdopen(fd, "w+")
        fo.write(data)
        fo.close()
        
    def read_json(self,path):
        flags = os.O_RDONLY
        modes = stat.S_IWUSR | stat.S_IRUSR
        with os.fdopen(os.open(path,flags,modes),'r',encoding = 'utf-8') as fin:
            file = json.load(fin)
        return file

    def dump_json(self,json_dict,output_folder):
        
        json_object = json.dumps(json_dict,indent = 4)
        filename = os.path.join(output_folder,"model_mapping.json")
        with open(filename, "w") as i :
            json.dump(json_object, i)
        return

    def read_file(self,location):
        print("Reading the input data")
        df = pd.read_csv(location,nrows = 10)
        return df

    def data_clean(self,df):
        df['cleaned_tweet'] = df['tweet_text'].replace(r'\'|\"|\,|\.|\?|\+|\-|\/|\=|\(|\)|\n|"', '', regex=True)
        df['cleaned_tweet'] = df['cleaned_tweet'].replace("  ", " ")
        
        df['cleaned_tweet'] = df['cleaned_tweet'].replace(r'<ed>','', regex = True)
        df['cleaned_tweet'] = df['cleaned_tweet'].replace(r'\B<U+.*>|<U+.*>\B|<U+.*>','', regex = True)
        
        # convert tweets to lowercase
        df['cleaned_tweet'] = df['cleaned_tweet'].str.lower()
        
        #remove user mentions
        #df['cleaned_tweet'] = df['cleaned_tweet'].replace(r'^(@\w+)',"", regex=True)
        
        #remove 'user' in the beginning
        df['cleaned_tweet'] = df['cleaned_tweet'].replace(r'(@user+)',"", regex=True)
        
        #remove_symbols
        df['cleaned_tweet'] = df['cleaned_tweet'].replace(r'[^a-zA-Z0-9]', " ", regex=True)
        
        #remove punctuations 
        df['cleaned_tweet'] = df['cleaned_tweet'].replace(r'[[]!"#$%\'()\*+,-./:;<=>?^_`{|}]+',"", regex = True)

        #remove_URL(x):
        df['cleaned_tweet'] = df['cleaned_tweet'].replace(r'https.*$', "", regex = True)

        #remove 'amp' in the text
        df['cleaned_tweet'] = df['cleaned_tweet'].replace(r'amp',"", regex = True)
        
        #remove words of length 1 or 2 
        df['cleaned_tweet'] = df['cleaned_tweet'].replace(r'\b[a-zA-Z]{1,2}\b','', regex=True)

        #remove extra spaces in the tweet
        df['cleaned_tweet'] = df['cleaned_tweet'].replace(r'^\s+|\s+$'," ", regex=True)
        
        #remove_digits
        df['cleaned_tweet'] = df['cleaned_tweet'].replace(r'[0-9]', "", regex=True)
        
        #remove stopwords and words_to_remove
        stop_words = set(stopwords.words('english'))
        mystopwords = [stop_words, "via", words_to_remove]
        
        df['fully_cleaned_tweet'] = df['cleaned_tweet'].apply(lambda x: ' '.join([word for word in x.split() if word not in mystopwords]))
        
        return df



    
    
    