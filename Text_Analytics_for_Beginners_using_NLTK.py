# Tokenization

## Sentence Tokenization

from nltk.tokenize import sent_tokenize

text = '''Hello Mr. Smith, how are you doing today? The weather is great, and the city is awesome.
The sky is pinkish-blue. You shouldn't eat carboard'''

tokenized_text = sent_tokenize(text)

print(tokenized_text)

## Word Tokenization

from nltk.tokenize import word_tokenize

tokenized_word = word_tokenize(text)

print(tokenized_word)

# Frequency Distribytion

from nltk.probability import FreqDist

fdist = FreqDist(tokenized_word)

print(fdist)

fdist.most_common(2)

## Frequency Distribution Plot

import matplotlib.pyplot as plt

fdist.plot(30, cumulative=False)

plt.show()

# Stopwords

from nltk.corpus import stopwords

stop_words=set(stopwords.words('english'))

print(stop_words)

## Removing Stopwords

tokenized_sent = word_tokenize(tokenized_text[0])

filtered_sent = []

for w in tokenized_sent:
	if w not in stop_words:
		filtered_sent.append(w)

print('Tokenized Sentence:', tokenized_sent)

print('Filtered Sentence:', filtered_sent)

# Lexicon Normalization

## Stemming

from nltk.stem import PorterStemmer

from nltk.tokenize import sent_tokenize, word_tokenize

ps = PorterStemmer()

stemmed_words = []

for w in filtered_sent:
    stemmed_words.append(ps.stem(w))

print('Filtered Sentence:', filtered_sent)

print('Stemmed Sentence:', stemmed_words)

## Lemmatization

from nltk.stem.wordnet import WordNetLemmatizer

lem = WordNetLemmatizer()

from nltk.stem.porter import PorterStemmer

stem = PorterStemmer()

word = 'flying'

print('Lemmatized Word:', lem.lemmatize(word, 'v'))

print('Stemmed Word:', stem.stem(word))

# POS Tagging
import nltk

sent = 'Albert Einstein was born in Ulm, Germany in 1879.'

tokens = nltk.word_tokenize(sent)

print(tokens)

nltk.pos_tag(tokens)

# Sentiment Analysis

## Performing Sentiment Analysis using Text Classification

import pandas as pd

# The dataset is available on Kaggle.
# You can download it from the following link: 
# https://www.kaggle.com/c/sentiment-analysis-on-movie-reviews/data

data = pd.read_csv('train.tsv', sep='\t')

data.head()

data.info()

data.Sentiment.value_counts()

Sentiment_count = data.groupby('Sentiment').count()

plt.bar(Sentiment_count.index.values, Sentiment_count['Phrase'])

plt.xlabel('Review Sentiments')

plt.ylabel('Number of Review')

plt.show()

# Feature Generation using Bag of Words (BoW)
# generate document term matrix (DTM) by using scikit-learn's CountVectorizer.

from sklearn.feature_extraction.text import CountVectorizer

from nltk.tokenize import RegexpTokenizer

# tokenizer to remove unwanted elements from out data like symbols and numbers

token = RegexpTokenizer(r'[a-zA-Z0-9]+')

cv = CountVectorizer(lowercase=True,
                     stop_words='english',
                     ngram_range=(1, 1),
                     tokenizer=token.tokenize)

text_counts = cv.fit_transform(data['Phrase'])

# Split train and test set

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(text_counts, data['Sentiment'], test_size=0.3, random_state=1)


# Model Building and Evaluation

# Build the Text Classification Model using CountVector(or BoW).

from sklearn.naive_bayes import MultinomialNB

# Import sklearn metrics modul for accuracy calculation

from sklearn import metrics

# Model generation using Multinomial Naive Bayes

clf = MultinomialNB().fit(X_train, y_train)

predicted = clf.predict(X_test)

print('MultinomialNB Accuracy:', metrics.accuracy_score(y_test, predicted))

# Build the Text Classification Model using TF-IDF.

from sklearn.feature_extraction.text import TfidfVectorizer

tf = TfidfVectorizer()

text_tf = tf.fit_transform(data['Phrase'])

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(text_tf, data['Sentiment'], test_size=0.3, random_state=123)

from sklearn.naive_bayes import MultinomialNB

from sklearn import metrics

clf = MultinomialNB().fit(X_train, y_train)

predicted = clf.predict(X_test)

print('MultinomialNB Accuracy:', metrics.accuracy_score(y_test, predicted))