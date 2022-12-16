import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import string
import nltk
import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)
from wordcloud import WordCloud,ImageColorGenerator
from PIL import Image
import urllib
import requests
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier

train = pd.read_csv('https://raw.githubusercontent.com/dD2405/Twitter_Sentiment_Analysis/master/train.csv')
train_original=train.copy()
print(train)

test=pd.read_csv('https://raw.githubusercontent.com/dD2405/Twitter_Sentiment_Analysis/master/test.csv')
original_test=test.copy()
print(test)

combine=train.append(test,ignore_index=True,sort=True)
combine.head(20)
combine.tail(20)

def remove_pattern(text,pattern):
  r=re.findall(pattern,text)
  for i in r:
    text=re.sub(i,"",text)
  return text
 
combine['Tidy_Tweets'] = np.vectorize(remove_pattern)(combine['tweet'],"@[\w]*")
combine['Tidy_Tweets']=combine['Tidy_Tweets'].str.replace("[^a-zA-Z#]"," ")
x_train_bow, x_valid_bow, y_train_bow, y_valid_bow = train_test_split(train_bow,train['label'],test_size=0.3,random_state=2)
x_train_tfidf, x_valid_tfidf, y_train_tfidf, y_valid_tfidf = train_test_split(train_tfidf_matrix,train['label'],test_size=0.3,random_state=17)

combine['Tidy_Tweets'] = combine['Tidy_Tweets'].apply(lambda x:' '.join([w for w in x.split() if len(w)>3]))
tokenized_tweet = combine['Tidy_Tweets'].apply(lambda x: x.split())

from nltk import PorterStemmer
ps = PorterStemmer()
tokenized_tweet = tokenized_tweet.apply(lambda x:[ps.stem(i) for i in x])
tokenized_tweet.head(10)
for i in range(len(tokenized_tweet)):
  tokenized_tweet[i] =  ' '.join(tokenized_tweet[i])

combine['Tidy_Tweets'] = tokenized_tweet
all_words_positive = ' '.join(text for text in combine['Tidy_Tweets'][combine['label']==0])

Mask=np.array(Image.open(requests.get('http://clipart-library.com/image_gallery2/Twitter-PNG-Image.png',stream=True).raw))
image_colors = ImageColorGenerator(Mask)
wc = WordCloud(background_color='black',height=1500,width=4000,mask=Mask).generate(all_words_positive)
plt.figure(figsize=(10,20))
plt.imshow(wc.recolor(color_func=image_colors),interpolation="hamming")
plt.axis('off')
plt.show()
all_words_negative = ' '.join(text for text in combine['Tidy_Tweets'][combine['label']==1])
Mask=np.array(Image.open(requests.get('http://clipart-library.com/image_gallery2/Twitter-PNG-Image.png',stream=True).raw))
image_colors = ImageColorGenerator(Mask)
wc = WordCloud(background_color='black',height=1500,width=4000,mask=Mask).generate(all_words_negative)
plt.figure(figsize=(10,20))
plt.imshow(wc.recolor(color_func=image_colors),interpolation="gaussian")
plt.axis('off')
plt.show()

def Hashtags_Extract(x):
  hashtags=[]

  for i in x:
    ht = re.findall(r'#(\w+)',i)
    hashtags.append(ht)
  return hashtags
 
ht_positive = Hashtags_Extract(combine['Tidy_Tweets'][combine['label']==0])
ht_positive_unnest = sum(ht_positive,[])
ht_negative = Hashtags_Extract(combine['Tidy_Tweets'][combine['label']==1])
ht_negative_unnest = sum(ht_negative,[])
 
word_freq_positive = nltk.FreqDist(ht_positive_unnest)
df_positive = pd.DataFrame({'Hashtags':list(word_freq_positive.keys()),'Count':list(word_freq_positive.values())})
df_positive_plot = df_positive.nlargest(20,columns='Count')
sns.barplot(data=df_positive_plot,y='Hashtags',x='Count')
sns.despine()

word_freq_negative = nltk.FreqDist(ht_negative_unnest)
word_freq_negative
df_negative = pd.DataFrame({'Hashtags':list(word_freq_negative.keys()),'Count':list(word_freq_negative.values())})
df_negative.head(10)
df_negative_plot = df_negative.nlargest(20,columns='Count')
sns.barplot(data=df_negative_plot,y='Hashtags',x='Count')
sns.despine()

from sklearn.feature_extraction.text import CountVectorizer

bow_vectorizer = CountVectorizer(max_df=0.90,min_df=2,max_features=1000,stop_words='english')
bow=bow_vectorizer.fit_transform(combine['Tidy_Tweets'])
df_bow = pd.DataFrame(bow.todense())
df_bow

from sklearn.feature_extraction.text import TfidfVectorizer

tfidf=TfidfVectorizer(max_df=0.90, min_df=2,max_features=1000,stop_words='english')
tfidf_matrix=tfidf.fit_transform(combine['Tidy_Tweets'])
df_tfidf = pd.DataFrame(tfidf_matrix.todense())
df_tfidf

train_bow = bow[:31962]
train_bow.todense()

train_tfidf_matrix = tfidf_matrix[:31962]
train_tfidf_matrix.todense()

Log_Reg = LogisticRegression(random_state=0,solver='lbfgs')
Log_Reg.fit(x_train_bow,y_train_bow)
prediction_bow = Log_Reg.predict_proba(x_valid_bow)
prediction_bow

# if prediction is greater than or equal to 0.3 than 1 else 0
# Where 0 is for positive sentiment tweets and 1 for negative sentiment tweets
prediction_int = prediction_bow[:,1]>=0.3

# converting the results to integer type
prediction_int = prediction_int.astype(np.int)
prediction_int

# calculating f1 score
log_bow = f1_score(y_valid_bow, prediction_int)
log_Ta=Log_Reg.score(x_train_bow,y_train_bow)
log_Va=Log_Reg.score(x_valid_bow,y_valid_bow)

print("F1 score: ",log_bow)
print("Training Accuracy: ",log_Ta)
print("Validation Accuracy: ",log_Va)

Log_Reg.fit(x_train_tfidf,y_train_tfidf)
prediction_tfidf = Log_Reg.predict_proba(x_valid_tfidf)
prediction_tfidf
prediction_int = prediction_tfidf[:,1]>=0.3

prediction_int = prediction_int.astype(np.int)
prediction_int

# calculating f1 score
log_tfidf = f1_score(y_valid_tfidf, prediction_int)
log_Ta_tfidf=Log_Reg.score(x_train_tfidf,y_train_tfidf)
log_Va_tfidf=Log_Reg.score(x_valid_tfidf,y_valid_tfidf)

print("F1 score: ",log_tfidf)
print("Training Accuracy: ",log_Ta_tfidf)

model_bow = XGBClassifier(random_state=22,learning_rate=0.9)
model_bow.fit(x_train_bow, y_train_bow)
xgb = model_bow.predict_proba(x_valid_bow)
# if prediction is greater than or equal to 0.3 than 1 else 0
# Where 0 is for positive sentiment tweets and 1 for negative sentiment tweets
xgb=xgb[:,1]>=0.3

# converting the results to integer type
xgb_int=xgb.astype(np.int)

# calculating f1 score
xgb_bow=f1_score(y_valid_bow,xgb_int)
xbg_Ta=model_bow.score(x_train_bow,y_train_bow)
xbg_Va=model_bow.score(x_valid_bow,y_valid_bow)

print("F1 score: ",xgb_bow)
print("Training Accuracy: ",xbg_Ta)
print("Validation Accuracy: ",xbg_Va)


model_tfidf = XGBClassifier(random_state=29,learning_rate=0.7)
model_tfidf.fit(x_train_tfidf, y_train_tfidf)
xgb_tfidf=model_tfidf.predict_proba(x_valid_tfidf)
# if prediction is greater than or equal to 0.3 than 1 else 0
# Where 0 is for positive sentiment tweets and 1 for negative sentiment tweets
xgb_tfidf=xgb_tfidf[:,1]>=0.3

# converting the results to integer type
xgb_int_tfidf=xgb_tfidf.astype(np.int)

# calculating f1 score
score=f1_score(y_valid_tfidf,xgb_int_tfidf)
score_Ta_tfidf=model_bow.score(x_train_tfidf,y_train_tfidf)
score_Va_tfidf=model_bow.score(x_valid_tfidf,y_valid_tfidf)

print("F1 score: ",score)
print("Training Accuracy: ",score_Ta_tfidf)
print("Validation Accuracy: ",score_Va_tfidf)

dct.fit(x_train_bow,y_train_bow)
dct_bow = dct.predict_proba(x_valid_bow)
# if prediction is greater than or equal to 0.3 than 1 else 0
# Where 0 is for positive sentiment tweets and 1 for negative sentiment tweets
dct_bow=dct_bow[:,1]>=0.3

# converting the results to integer type
dct_int_bow=dct_bow.astype(np.int)

# calculating f1 score
dct_score_bow=f1_score(y_valid_bow,dct_int_bow)
dct_Ta=dct.score(x_train_bow,y_train_bow)
dct_Va=dct.score(x_valid_bow,y_valid_bow)

print("F1 score: ",dct_score_bow)
print("Training Accuracy: ",dct_Ta)
print("Validation Accuracy: ",dct_Va)

dct.fit(x_train_tfidf,y_train_tfidf)
dct_tfidf = dct.predict_proba(x_valid_tfidf)

dct_tfidf=dct_tfidf[:,1]>=0.3

# converting the results to integer type
dct_int_tfidf=dct_tfidf.astype(np.int)

# calculating f1 score
dct_score_tfidf=f1_score(y_valid_tfidf,dct_int_tfidf)
dct_Ta_tfidf=dct.score(x_train_tfidf,y_train_tfidf)
dct_Va_tfidf=dct.score(x_valid_tfidf,y_valid_tfidf)
                       

print("F1 score: ",dct_score_tfidf)
print("Training Accuracy: ",dct_Ta_tfidf)
print("Validation Accuracy: ",dct_Va_tfidf)


Algo_1 = ['LogisticRegression(Bag-of-Words)','XGBoost(Bag-of-Words)','DecisionTree(Bag-of-Words)']

score_1 = [log_bow,xgb_bow,dct_score_bow]

score_11 = [log_Ta,xbg_Ta,dct_Ta,]

score_12 = [log_Va,xbg_Va,dct_Va]

compare_1 = pd.DataFrame({'Model':Algo_1,'F1_Score':score_1,'Training_Accuracy':score_11,'Validation_Accuracy':score_12},index=[i for i in range(1,4)])

compare_1.T

plt.figure(figsize=(18,5))

sns.pointplot(x='Model',y='F1_Score',data=compare_1)

plt.title('Bag-of-Words')
plt.xlabel('MODEL')
plt.ylabel('SCORE')

plt.show()

plt.figure(figsize=(18,5))

sns.pointplot(x='Model',y='Training_Accuracy',data=compare_1)

plt.title('Bag-of-Words')
plt.xlabel('MODEL')
plt.ylabel('SCORE')

plt.show()

plt.figure(figsize=(18,5))

sns.pointplot(x='Model',y='Validation_Accuracy',data=compare_1)

plt.title('Bag-of-Words')
plt.xlabel('MODEL')
plt.ylabel('SCORE')

plt.show()

Algo_2 = ['LogisticRegression(TF-IDF)','XGBoost(TF-IDF)','DecisionTree(TF-IDF)']

score_2 = [log_tfidf,score,dct_score_tfidf]

score_21 = [log_Ta_tfidf,score_Ta_tfidf,dct_Ta_tfidf]

score_22 = [log_Va_tfidf,score_Va_tfidf,dct_Va_tfidf]

compare_2 = pd.DataFrame({'Model':Algo_2,'F1_Score':score_2,'Training_Accuracy':score_21,'Validation_Accuracy':score_22},index=[i for i in range(1,4)])

compare_2.T

plt.figure(figsize=(18,5))

sns.pointplot(x='Model',y='F1_Score',data=compare_2)

plt.title('TF-IDF')
plt.xlabel('MODEL')
plt.ylabel('SCORE')

plt.show()

plt.figure(figsize=(18,5))

sns.pointplot(x='Model',y='Training_Accuracy',data=compare_2)

plt.title('TF-IDF')
plt.xlabel('MODEL')
plt.ylabel('SCORE')

plt.show()

plt.figure(figsize=(18,5))

sns.pointplot(x='Model',y='Validation_Accuracy',data=compare_2)

plt.title('TF-IDF')
plt.xlabel('MODEL')
plt.ylabel('SCORE')

plt.show()

Algo_best = ['LogisticRegression(Bag-of-Words)','LogisticRegression(TF-IDF)']

score_best = [log_bow,log_tfidf]

compare_best = pd.DataFrame({'Model':Algo_best,'F1_Score':score_best},index=[i for i in range(1,3)])

compare_best.T

plt.figure(figsize=(18,5))

sns.pointplot(x='Model',y='F1_Score',data=compare_best)

plt.title('Logistic Regression(Bag-of-Words & TF-IDF)')
plt.xlabel('MODEL')
plt.ylabel('SCORE')

plt.show()

test_tfidf = tfidf_matrix[31962:]
test_pred = Log_Reg.predict_proba(test_tfidf)

test_pred_int = test_pred[:,1] >= 0.3
test_pred_int = test_pred_int.astype(np.int)

test['label'] = test_pred_int

submission = test[['id','label']]
submission.to_csv('result.csv', index=False)

pd.read_csv('result.csv')
sns.countplot(train_original['label'])

