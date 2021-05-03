import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
import string
import tensorflow as tf
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from sklearn.metrics import classification_report, confusion_matrix


def top_N_tags(df, N=10):
    tag_list = []
    for tweet in df.index:
        tag_list += re.findall(r'@\w+', df.loc[tweet, "OriginalTweet"])
    
    tag_sort = pd.DataFrame(tag_list)
    tag_sort = tag_sort.groupby(0).size().sort_values(ascending=False)
    return tag_sort.head(N).index.to_list()
    
def top_N_hashtags(df, N=10):
    hashtag_list = []
    for tweet in df.index.values:
        # Exclude #covid* and #corova* because those tags were the criteria
        # used to collect data and hence in most of the tweets.
        hashtags = re.findall(r'#\w+', df.loc[tweet, "OriginalTweet"])
        hashtag_list += [hashtag for hashtag in hashtags 
                         if "covid" not in hashtag.lower()
                         and "corona" not in hashtag.lower()]

    hashtag_sort = pd.DataFrame(hashtag_list)
    hashtag_sort = hashtag_sort.groupby(0).size().sort_values(ascending=False)
    return hashtag_sort.head(N).index.to_list()
    
def remove_link(text):
    return re.sub(r'http\S+', " ", text)

def remove_tag(text, top_tags):
    keep_tags = []
    for tag in top_tags:
        keep_tags += re.findall(tag, text)
    text = re.sub(r'@\S+', " ", text)
    return " ".join(keep_tags + [text])
    
def remove_hashtag(text, top_hashtags):
    keep_hashtags = []
    for tag in top_hashtags:
        keep_hashtags += re.findall(tag, text)
    text = re.sub(r'#\S+', " ", text)
    return " ".join(keep_hashtags + [text])
    
def remove_special_char(text):
    for special_ch in (string.punctuation + string.digits + string.whitespace):
        text = text.replace(special_ch, " ")
    return text

def remove_non_english(text):
    clean_text = [word for word in text.split() if word.isalpha()]
    return " ".join(clean_text)

def remove_stop_words(text, stop_words):
    clean_text = [word for word in text.split() if word not in stop_words]
    return " ".join(clean_text)

def stemming(text, ps):
    clean_text = [ps.stem(word) for word in text.split()]
    return " ".join(clean_text)

def lemmatization(text, wnl):
    clean_text = [wnl.lemmatize(word) for word in text.split()]
    return " ".join(clean_text)
    
def plot_tweet_len(df):
    len_tweet = [len(tweet.split()) for tweet in df.CleanTweet]
    plt.figure()
    plt.hist(len_tweet, bins=20)
    plt.title("Histogram of length of tweets")    

def fit_tokenizer(df, num_words=15000):
    tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=num_words, oov_token="<OOV>")
    tokenizer.fit_on_texts(df.CleanTweet.values)
    return tokenizer

def tokenize_text(tokenizer, df, max_len):
    sequences = tokenizer.texts_to_sequences(df.CleanTweet.values)
    corpus = tf.keras.preprocessing.sequence.pad_sequences(sequences, maxlen=max_len, truncating='post', padding='post')
    return corpus
    
def evaluate(model, test_corpus, test_label):
    test_predict = np.argmax(model.predict(test_corpus), axis=1)
    
    # Accuracy
    dict(zip(model.metrics_names, model.evaluate(test_corpus, test_label, verbose=2)))

    # Confusion matrix
    print(confusion_matrix(test_label, test_predict))

    # Precision, Recall, and F1 score
    print(classification_report(test_label, test_predict, digits=3))

