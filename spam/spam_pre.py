import numpy as np
import pandas as pd
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer

# Run the below piece of code for the first time
# nltk.download('stopwords')
# Importing Data

message_data = pd.read_csv("spam.csv", encoding="latin")
message_data = message_data.drop(
    ['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis=1)
message_data = message_data.rename(
    columns={'v1': 'Spam/Not_Spam', 'v2': 'message'})
message_data.groupby('Spam/Not_Spam').describe()
message_data_copy = message_data['message'].copy()


def text_preprocess(text):
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = [word for word in text.split() if word.lower()
            not in stopwords.words('english')]
    return " ".join(text)


message_data_copy = message_data_copy.apply(text_preprocess)


vectorizer = TfidfVectorizer()
message_mat = vectorizer.fit_transform(message_data_copy)

# Split data into training and testing
message_train, message_test, spam_nospam_train, spam_nospam_test = train_test_split(message_mat,
                                                                                    message_data['Spam/Not_Spam'], test_size=0.3, random_state=20)


Spam_model = LogisticRegression(solver='liblinear', penalty='l1')
Spam_model.fit(message_train, spam_nospam_train)
pred = Spam_model.predict(message_test)
print("accuracy_score:", accuracy_score(spam_nospam_test, pred))

##########################################

# # stemming


def stemmer(text):
    text = text.split()
    words = ""
    for i in text:
        stemmer = SnowballStemmer("english")
        words += (stemmer.stem(i))+" "
    return words


message_data_copy = message_data_copy.apply(stemmer)
vectorizer = TfidfVectorizer()
message_mat = vectorizer.fit_transform(message_data_copy)

message_train, message_test, spam_nospam_train, spam_nospam_test = train_test_split(message_mat,
                                                                                    message_data['Spam/Not_Spam'], test_size=0.3, random_state=20)


Spam_model = LogisticRegression(solver='liblinear', penalty='l1')
Spam_model.fit(message_train, spam_nospam_train)
pred = Spam_model.predict(message_test)
print("accuracy_score after stemming:", accuracy_score(spam_nospam_test, pred))

#######################################################
# normalizing length.
message_data['length'] = message_data['message'].apply(len)
message_data.head()
length = message_data['length'].to_numpy()
new_mat = np.hstack((message_mat.todense(), length[:, None]))
message_train, message_test, spam_nospam_train, spam_nospam_test = train_test_split(new_mat,
                                                                                    message_data['Spam/Not_Spam'], test_size=0.3, random_state=20)
Spam_model = LogisticRegression(solver='liblinear', penalty='l1')
Spam_model.fit(message_train, spam_nospam_train)
pred = Spam_model.predict(message_test)
print("accuracy_score after normalization:",
      accuracy_score(spam_nospam_test, pred))
