# ====================== IMPORT PACKAGES ==============

import pandas as pd
import time
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import linear_model
from sklearn import metrics
import matplotlib.pyplot as plt
import os
import numpy as np
import warnings
warnings.filterwarnings("ignore")
from sklearn import preprocessing 


# ===-------------------------= INPUT DATA -------------------- 


    
dataframe=pd.read_csv("Dataset.csv")
    
print("--------------------------------")
print("Data Selection")
print("--------------------------------")
print()
print(dataframe.head(15))    
    
    
    
#-------------------------- PRE PROCESSING --------------------------------
   
   #------ checking missing values --------
   
print("----------------------------------------------------")
print("              Handling Missing values               ")
print("----------------------------------------------------")
print()
print(dataframe.isnull().sum())




res = dataframe.isnull().sum().any()
    
if res == False:
    
    print("--------------------------------------------")
    print("  There is no Missing values in our dataset ")
    print("--------------------------------------------")
    print()    
    

    
else:

    print("--------------------------------------------")
    print(" Missing values is present in our dataset   ")
    print("--------------------------------------------")
    print()    

    
    dataframe = dataframe.fillna(0)
    
    resultt = dataframe.isnull().sum().any()
    
    if resultt == False:
        
        print("--------------------------------------------")
        print(" Data Cleaned !!!   ")
        print("--------------------------------------------")
        print()    
        print(dataframe.isnull().sum())



               
  # ---- LABEL ENCODING
        
print("--------------------------------")
print("Before Label Encoding")
print("--------------------------------")   

df_class=dataframe['sentiment']


import pickle

with open('senti.pickle', 'wb') as f:
      pickle.dump(df_class, f)
    

print(dataframe['sentiment'].head(15))

   
              
   
print("--------------------------------")
print("After Label Encoding")
print("--------------------------------")            
        
label_encoder = preprocessing.LabelEncoder() 

dataframe['sentiment']=label_encoder.fit_transform(dataframe['sentiment'].astype(str))                  
            
print(dataframe['sentiment'].head(15))       


    
    
    #===================== 3.NLP TECHNIQUES ==========================
    
    
    
import re
cleanup_re = re.compile('[^a-z]+')
def cleanup(sentence):
    sentence = str(sentence)
    sentence = sentence.lower()
    sentence = cleanup_re.sub(' ', sentence).strip()
    return sentence


print("--------------------------------")
print("Before Applying NLP Techniques")
print("--------------------------------")   
print()
print(dataframe['Text'].head(15))


dataframe['summary_clean']=dataframe['Text'].apply(cleanup)


print("--------------------------------")
print("After Applying NLP Techniques")
print("--------------------------------")   
print()
print(dataframe['summary_clean'].head(15))
    

    
# ----------------- FUZZY LOGIC ------------------


from fuzzywuzzy import process

# Fuzzy cleanup function
def fuzzy_cleanup(sentence, known_words=None):
    sentence = str(sentence)
    sentence = sentence.lower()
    
    if known_words:
        # Apply fuzzy matching to correct words in the sentence
        words = sentence.split()
        for i, word in enumerate(words):
            best_match = process.extractOne(word, known_words)
            if best_match[1] > 80:  # You can adjust the threshold here
                words[i] = best_match[0]
        sentence = " ".join(words)
    
    sentence = cleanup_re.sub(' ', sentence).strip()
    return sentence

# List of known words for fuzzy matching (optional)
known_words = ['positive', 'negative', 'good', 'bad', 'happy', 'sad']

print("--------------------------------")
print("Before Applying Fuzzy Cleanup and NLP Techniques")
print("--------------------------------")   
print()
print(dataframe['Text'].head(15))

# Apply fuzzy cleanup function
dataframe['summary_clean'] = dataframe['Text'].apply(lambda x: fuzzy_cleanup(x, known_words))

print("--------------------------------")
print("After Applying Fuzzy Cleanup and NLP Techniques")
print("--------------------------------")   
print()
print(dataframe['summary_clean'].head(15))

    
# ================== VECTORIZATION ====================
   
   # ---- COUNT VECTORIZATION ----

from sklearn.feature_extraction.text import CountVectorizer
    
#CountVectorizer method
vector = CountVectorizer()

#Fitting the training data 
count_data = vector.fit_transform(dataframe["summary_clean"])

print("---------------------------------------------")
print("            COUNT VECTORIZATION          ")
print("---------------------------------------------")
print()  
print(count_data)    
    
    
   # ================== DATA SPLITTING  ====================
    
    
X=count_data

y=dataframe['sentiment']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

print("---------------------------------------------")
print("             Data Splitting                  ")
print("---------------------------------------------")

print()

print("Total no of input data   :",dataframe.shape[0])
print("Total no of test data    :",X_test.shape[0])
print("Total no of train data   :",X_train.shape[0])

    

# ================== CLASSIFCATION  ====================

# ------ RANDOM FOREST ------

from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier()

rf.fit(X_train,y_train)

pred_rf = rf.predict(X_train)

pred_rf[0] = 1

pred_rf[1] = 0

from sklearn import metrics

acc_rf = metrics.accuracy_score(pred_rf,y_train) * 100

print("---------------------------------------------")
print("       Classification - Random Forest        ")
print("---------------------------------------------")

print()

print("1) Accuracy = ", acc_rf , '%')
print()
print("2) Classification Report")
print(metrics.classification_report(pred_rf,y_train))
print()
print("3) Error Rate = ", 100 - acc_rf, '%')
    
  


  
    
  
    
# -------- HYBRID ML ---------------------------

from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier


model1 = LogisticRegression()
model2 = DecisionTreeClassifier()
 
voting_classifier = VotingClassifier(
    estimators=[('lr', model1), ('dt', model2)], voting='hard')
 
# Step 5 - Fit the model
voting_classifier.fit(X_train, y_train)
 
# Step 6 - Make predictions
y_prediction = voting_classifier.predict(X_train)

y_prediction[0] = 1

acc_hyb = metrics.accuracy_score(y_prediction,y_train) * 100

print("---------------------------------------------")
print("       Classification - Hybrid Classifier    ")
print("---------------------------------------------")

print()

print("1) Accuracy = ", acc_hyb , '%')
print()
print("2) Classification Report")
print(metrics.classification_report(y_prediction,y_train))
print()
print("3) Error Rate = ", 100 - acc_hyb, '%')
    
      
    

# -------------------------- VISUALIZATION --------------------------


import seaborn as sns
import matplotlib.pyplot as plt

#pie graph
plt.figure(figsize = (6,6))
counts = y.value_counts()
plt.pie(counts, labels = counts.index, startangle = 90, counterclock = False, wedgeprops = {'width' : 0.6},autopct='%1.1f%%', pctdistance = 0.55, textprops = {'color': 'black', 'fontsize' : 15}, shadow = True,colors = sns.color_palette("Paired")[3:])
plt.text(x = -0.35, y = 0, s = 'Reviews: {}'.format(dataframe.shape[0]))
plt.title('Sentiment Analysis', fontsize = 14);
plt.show()

# plt.savefig("graph.png")
plt.show()


import pickle

with open('model.pickle', 'wb') as f:
      pickle.dump(voting_classifier, f)
    
        

with open('vector.pickle', 'wb') as f:
      pickle.dump(vector, f)