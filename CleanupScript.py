import numpy as np
import pandas as pd
import re
import string
import matplotlib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier,RidgeClassifier,PassiveAggressiveClassifier
import pickle
from sklearn.metrics import precision_recall_fscore_support,accuracy_score
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib
from datetime import datetime
import matplotlib.pyplot as plt

from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix 
%matplotlib inline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
import collections as ct
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from nltk.corpus import stopwords

data_filepath = "./finalout_less_new.csv"
input_df = pd.read_csv('finalout_less_new.csv')
print(input_df.shape)
cols = ['Resolution','Subject',
       'Symptom','ActualService',
       'OwnerTeam']
label_freq_threshold = 10
#input_df = input_df[cols]


#Harshit Code 
from nltk.corpus import stopwords
special_chars = string.punctuation  
stop = stopwords.words('english')
from datetime import datetime
import collections as ct
from nltk.corpus import stopwords

def junk_clean(text):
    
#     starttime = datetime.now()
#     tagged = ne_chunk(pos_tag(word_tokenize(text)))
#     person = []
#     for subtree in tagged.subtrees(filter=lambda t: t.label() == 'PERSON'):
#         for leaf in subtree.leaves():
#             person.append(leaf[0])
#     print("Person Tagging: " + (datetime.now() - starttime))
    
    starttime1 = datetime.now()
    junk = []
    body = []
    b = text.splitlines()
    for index, i in enumerate(b):
        c = i.split()
        if len(c) > 0:
            sp = sum(v for k, v in ct.Counter(i).items() if k in special_chars)/len(c)
            #name_perc = len(list(set(c).intersection(person)))/len(c)
            stop_perc = len(list(set(c).intersection(stop)))/len(c)
            if len(c) < 10  or sp >= 0.5 or stop_perc <= 0.05:
                junk.append(i)
                #print(index, "\t", "%.2f" % name_perc, "\t",len(c),"\t","%.2f" % sp ,"\t","%.2f" % stop_perc, "\t", i)
            else:
                body.append(i)
        else:
            continue
            
    #print("Person Tagging: %s" % (datetime.now() - starttime1))
    
    junk = '\n'.join(junk)
    body = '\n'.join(body)
    return pd.Series({'junk': junk, 'body': body})

def parse_text(text):
    for pattern, replacement in new_dict_comp.items():
        text = pattern.sub(replacement, text)
    return text

def remove_signatures(temp):
    i = 50;
    startTime1 = datetime.now()
    temp["New_Symptom"] = temp["body"];
    for i in range(300,9,-1):
        for j in range(10,9,-1):
            try:
                pattern = "[\\S]+";
                tf_vectorizer = CountVectorizer(min_df=j,
                                                ngram_range = (i,i),
                                                analyzer='word',
                                                tokenizer=lambda x: x.split(' '),
                                                lowercase = False
                                                )
                tf_vectorizer.fit(temp["New_Symptom"])
                startTime3 = datetime.now()
                a =  zip(tf_vectorizer.get_feature_names(), itertools.repeat(""))
                new_dict = {re.escape(k): v for k, v in a}
                new_dict_comp = {re.compile(k): v for k, v in new_dict.items()}
                temp["New_Symptom"] = temp["New_Symptom"].apply(parse_text)
                return(temp)
            except ValueError:
                print(ValueError)
            except Exception as e:
                print(e)


#Harshit Code Ends

def remove_and_dedup_punctuation_numbers(text): 
    result = text.replace('[\w\d\.-]+@[\w\d\.-]+',' ').replace(
    '\d+', ' ').replace('_+', ' ').replace('+', ' ').replace(
    '.', ' ').replace('^https?:\/\/.*[\r\n]*', '').replace('[^\w]',' ').lower()
    result = re.sub (r'([^a-zA-Z\s.?!])', '', result)
    result = re.sub(r"([" + re.escape(string.punctuation) + r"])\1+", r"\1", result)
    result = re.sub( '\s+', ' ', result ).strip()
    return result

def clean_text(row):
    sub = row['Subject']
    sym = row['Symptom']
    res = row['Resolution']
    if row['Subject'] is None:
        sub = ''
    if row['Symptom'] is None:
        sym = ''
    if row['Resolution'] is None:
        res = ''
    train_text = sub+" "+sym+" "+res
    text_clean = remove_and_dedup_punctuation_numbers(train_text)
    row["Text_Train_Cleaned"] = text_clean
    
    test_text = sub+" "+sym
    text_clean = remove_and_dedup_punctuation_numbers(test_text)
    row["Text_Test_Cleaned"] = text_clean
    
    return row

def encode_decision(y):
    encoder= LabelEncoder()
    encoder.fit(y)
    y_encoded=encoder.transform(y)
    return y_encoded,encoder


def clean_df(df):
    truncated_df_null_removed = df[cols].dropna()
    #print(len(truncated_df_null_removed))
    df_cleaned = truncated_df_null_removed.apply(clean_text,1)[["Text_Train_Cleaned","Text_Test_Cleaned","OwnerTeam","ActualService"]]
    #print(len(df_cleaned))
    training_dataframe = df_cleaned.groupby(["ActualService","OwnerTeam"]).filter(lambda x: len(x) > label_freq_threshold)
    #print(len(training_dataframe))
    return training_dataframe

def create_train_test_df(df):
    y = df[["ActualService","OwnerTeam"]].as_matrix()
    train_dataframe, test_dataframe = train_test_split(df,test_size=0.2,stratify=y,random_state=0)
    return train_dataframe,test_dataframe


input_df = pd.read_csv(data_filepath)
input_df = input_df[cols]
input_df['Symptom'] = input_df['Symptom'].astype(str)
#new = input_df["Symptom"].apply(junk_clean)

new = input_df["Symptom"].astype(str).apply(junk_clean)
input_df = pd.concat([input_df, new], axis=1)
sum(input_df["body"]!="")
#temp = pd.concat([input_df, new], axis=1)


input_df["body"]  = np.where(input_df["body"] == "",input_df['junk'],input_df["body"])

input_df['body'] = input_df['body'].replace(regex='(\\s*\\n\\s*)+',value='.')
input_df['body'] = input_df['body'].replace(regex='(\\s*\\r\\s*)+',value='.')
input_df['body'] = input_df['body'].replace(regex='(\\s*\\t\\s*)+',value='.')
input_df['body'] = input_df['body'].replace(regex='(\\s*\\-\\s*)+',value='-')


import re
from datetime import datetime
import itertools
def countTotal(ser,vecs):
    print(len(vecs))
    total = 0;
    for vec in vecs:
        a = ser.str.contains(vec).sum()
        print(a)
        #print(ser)
        #print(ser.str.contains("original"))
        total += a
    print("Total occurances - %d" %total)
    
def parse_text(text):
    for pattern, replacement in new_dict_comp.items():
        text = pattern.sub(replacement, text)
    return text

i = 50;
startTime1 = datetime.now()
input_df["New_Symptom"] = input_df["body"];
for i in range(300,9,-1):
    for j in range(10,9,-1):
        startTime = datetime.now()
        startTime2 = datetime.now()
        
        print(i , j)
        try:
            pattern = "[\\S]+";
            tf_vectorizer = CountVectorizer( min_df=j,
                                            ngram_range = (i,i),
                                            analyzer='word',
                                            tokenizer=lambda x: x.split(' '),
                                            lowercase = False
                                            )
            tf_vectorizer.fit(input_df["New_Symptom"])
            print(tf_vectorizer.get_feature_names())
            print("total vector size - %d" %(len(tf_vectorizer.get_feature_names())))
            
            startTime3 = datetime.now()
            
            a =  zip(tf_vectorizer.get_feature_names(), itertools.repeat(""))
            new_dict = {re.escape(k): v for k, v in a}
            new_dict_comp = {re.compile(k): v for k, v in new_dict.items()}
#             countTotal(input_df["New_Symptom"],tf_vectorizer.get_feature_names())
            input_df["New_Symptom"] = input_df["New_Symptom"].apply(parse_text)
            
#             for vec in tf_vectorizer.get_feature_names():
#                 #print(input_df["New_Symptom"].str.contains(vec).sum())
#                 input_df["New_Symptom"] = input_df["New_Symptom"].str.replace(re.escape(vec),"")  

            print("time to replace - ")
            print(datetime.now() - startTime3)
            #countTotal(input_df["New_Symptom"],tf_vectorizer.get_feature_names())
#             pattern = "[\\S]+";
#             tf_vectorizer = CountVectorizer( min_df=j,
#                                             ngram_range = (i,i),
#                                             analyzer='word',
#                                             tokenizer=lambda x: x.split(' '),
#                                             lowercase = False
#                                             )
#             tf_vectorizer.fit(temp["New_Symptom"])
#             print("total vector size after replacement - %d" %(len(tf_vectorizer.get_feature_names())))
#             #print(tf_vectorizer.get_feature_names()[0])
#             #input_df["New_Symptom"] = temp
#             print("time to vectorize 2 times- ") 
#             print(datetime.now() - startTime2)
        except ValueError:
            print(ValueError)
        except Exception as e:
            print(e)
        print(datetime.now() - startTime)
print("Total Time: - ")
print(datetime.now() - startTime1)





