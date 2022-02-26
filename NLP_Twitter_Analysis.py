import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Load the data
data=pd.read_csv('twitter.csv')

data.info()
data

data.describe()

data['tweet']

tweets_df=data

# Drop the 'id' column
tweets_df = tweets_df.drop(['id'], axis=1)

 sns.heatmap(tweets_df.isnull(), yticklabels = False, cbar = False, cmap="Blues")

 data.hist(bins = 30, figsize = (13,5), color = 'r')

 sns.countplot(tweets_df['label'], label = "Count")

 # Let's get the length of the messages
tweets_df['length']=tweets_df['tweet'].apply(len)

tweets_df['length'].plot(bins=100, kind='hist')

tweets_df.describe()

# Let's see the shortest message 
tweets_df[tweets_df['length'] == 11]['tweet'].iloc[0]

# Let's see the message with mean length 
tweets_df[tweets_df['length'] == 84]['tweet'].iloc[0]

positive = tweets_df[tweets_df['label']==0]

positive

negative = tweets_df[tweets_df['label']==1]

negative

sentences=tweets_df['tweet'].tolist()

sentences

len(sentences)

sentences_as_one_string = " ".join(sentences)

import string
string.punctuation

Test = 'Good morning beautiful people :)... I am having fun learning Machine learning and AI!!'

Test_punc_removed=[char for char in Test if char not in string.punctuation]

# Join the characters again to form the string.
Test_punc_removed_join=' '.join(Test_punc_removed)
Test_punc_removed_join

Test_punc_removed = []
for char in Test: 
    if char not in string.punctuation:
        Test_punc_removed.append(char)
        
# Join the characters again to form the string.
Test_punc_removed_join = ''.join(Test_punc_removed)
Test_punc_removed_join

import nltk # Natural Language tool kit 
nltk.download('stopwords')

# You have to download stopwords Package to execute this command
from nltk.corpus import stopwords
stopwords.words('english')

Test_punc_removed_join_clean=[word for word in Test_punc_removed_join.split()  if word.lower() not in stopwords.words('english')]

Test_punc_removed_join_clean # Only important (no so common) words are left

mini_challenge = 'Here is a mini challenge, that will teach you how to remove stopwords and punctuations!'

challege = [ char     for char in mini_challenge  if char not in string.punctuation ]
challenge = ''.join(challege)
challenge = [  word for word in challenge.split() if word.lower() not in stopwords.words('english')  ]

from sklearn.feature_extraction.text import CountVectorizer
sample_data = ['This is the first paper.','This document is the second paper.','And this is the third one.','Is this the first paper?']

vectorizer= CountVectorizer()
X=vectorizer.fit_transform(sample_data)

print(vectorizer.get_feature_names())

print(X.toarray())

mini_challenge = ['Hello World','Hello Hello World','Hello World world world']

vectorizer_challenge = CountVectorizer()
X_challenge = vectorizer_challenge.fit_transform(mini_challenge)
print(X_challenge.toarray())

# Let's define a pipeline to clean up all the messages 
# The pipeline performs the following: (1) remove punctuation, (2) remove stopwords

def message_cleaning(message):
    Test_punc_removed = [char for char in message if char not in string.punctuation]
    Test_punc_removed_join = ''.join(Test_punc_removed)
    Test_punc_removed_join_clean = [word for word in Test_punc_removed_join.split() if word.lower() not in stopwords.words('english')]
    return Test_punc_removed_join_clean

# Let's test the newly added function
tweets_df_clean = tweets_df['tweet'].apply(message_cleaning)

print(tweets_df_clean[5]) # show the cleaned up version

print(tweets_df['tweet'][5]) # show the original version

from sklearn.feature_extraction.text import CountVectorizer
# Define the cleaning pipeline we defined earlier
vectorizer = CountVectorizer(analyzer = message_cleaning, dtype = np.uint8)
tweets_countvectorizer = vectorizer.fit_transform(tweets_df['tweet'])

print(vectorizer.get_feature_names())

print(tweets_countvectorizer.toarray())

tweets_countvectorizer.shape

tweets = pd.DataFrame(tweets_countvectorizer.toarray())

X=tweets

X

y = tweets_df['label']

X.shape

y.shape

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

from sklearn.naive_bayes import MultinomialNB

NB_classifier = MultinomialNB()
NB_classifier.fit(X_train, y_train)

from sklearn.metrics import classification_report, confusion_matrix

# Predicting the Test set results
y_predict_test = NB_classifier.predict(X_test[0:6393])
cm = confusion_matrix(y_test, y_predict_test)
sns.heatmap(cm, annot=True)

print(classification_report(y_test, y_predict_test))






