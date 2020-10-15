# -*- coding: utf-8 -*-
"""
This file solves COVID tweet sentiment classification problem.
"""

import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
import nltk
import json
import requests
import re
import string


class Log:
    # Log helper class.

    LOG_LEVEL = dict((value, index) for (index, value) in 
                     enumerate(["INFO", "ERROR", "FATEL"]))
    LEVEL = LOG_LEVEL["INFO"]
    
    @classmethod
    def set_log_level(cls, level="INFO"):
        """Set log level.

        Args:
            level (str, optional): level to be set. Defaults to "INFO".
            
        """
        cls.LEVEL = cls.LOG_LEVEL[level]
    
    @classmethod
    def print_msg(cls, level, msg):
        """Check log level and print permissible logs.

        Args:
            level (str): level of the message.
            msg (str): message to be print.
            
        """
        if cls.LOG_LEVEL[level] >= cls.LEVEL:
            print(msg)


class Common:
    # Dictionaries and lists

    @staticmethod
    def lemmatization_dict():
        # Dictionary of words for lemmatization.
        return {"shop":"store", "supermarket":"market", "paper":"toiletpaper",
            "much":"many", "employee":"worker", "staff":"worker",
            "global":"world", "company":"business", "consumer":"customer", 
            "house":"home", "grocery":"goods", "products":"goods", 
            "toilet":"toiletpaper"}
    
    @staticmethod
    def stemming_dict():
        # Dictionary of words for stemming.
        return {"buying":"buy", "bought":"buy", "working":"work", 
            "worked":"work", "shopping":"shop", "shopped":"shop", 
            "shops":"shop", "days":"day", "weeks":"week", "masks":"mask", 
            "got":"get", "gets":"get", "consumers":"consumer", "gets":"get", 
            "supermarkets":"supermarket", "says":"say", "saying":"say", 
            "said":"say", "getting":"get", "companies":"company", "went":"go", 
            "got":"get", "making":"make", "made":"make", "services":"service", 
            "hours":"hour", "years":"year", "products":"product", "going":"go", 
            "increases":"increase", "increased":"increase", "markets":"market", 
            "close":"closed", "needs":"need", "customers":"customer", 
            "stores":"store", "businesses":"business", "employees":"employee", 
            "workers":"worker", "staffs":"staff", "needed":"need",
            "hands":"hand", }
    
    @staticmethod
    def stopword_list():
        # List of stopwords.
        nltk.download('stopwords')
        stopwords = list(set(nltk.corpus.stopwords.words('english')))
        stopwords += ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "l", 
                      "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", 
                      "x", "y", "z"]
        stopwords += ["in", "on", "at", "via", "due", "could", "would", "may", 
                      "one", "still", "even", "also", "every", "two", "etc", 
                      "per"]
        stopwords += ["covid", "corona", "virus"]
        return stopwords
    
    @staticmethod
    def create_loc_dict():
        # Dictionary of locations.
        loc_dict = {}
        states = json.loads(requests.get("https://raw.githubusercontent.com" \
            "/praneshsaminathan/country-state-city/master/"\
            "states.json").text)["states"]
        
        countries=json.loads(requests.get("https://raw.githubusercontent.com"\
            "/praneshsaminathan/country-state-city/master/"\
            "countries.json").text)["countries"]
        
        cities = json.loads(requests.get("https://raw.githubusercontent.com/"\
            "praneshsaminathan/country-state-city/master/"\
            "cities.json").text)["cities"]
            
        us_states = pd.read_csv("https://worldpopulationreview.com/static/"\
            "states/abbr-name.csv",names=['state_code','state'])
        
        # Assign all cities its country code
        for city in cities:
            state_id = int(re.findall(r'\d+', city["state_id"])[0])
            country_id = states[state_id-1]["country_id"]
            country_id = int(re.findall(r'\d+', country_id)[0])
            country_name = countries[country_id-1]["sortname"]
            loc_dict[city["name"].lower()] = country_name.lower()
        
        # Assign all states its country code
        for state in states:
            country_id = int(re.findall(r'\d+', state["country_id"])[0])
            country_name = countries[country_id-1]["sortname"]
            loc_dict[state["name"].lower()] = country_name.lower()
        
        # Assign all countries its country code
        for country in countries:    
            loc_dict[country["sortname"].lower()] = country["sortname"].lower()
            loc_dict[country["name"].lower()] = country["sortname"].lower()
        
        # Assign "us" to all US locations
        for index in us_states.index:
            state = us_states.loc[index]
            loc_dict[state.state_code.lower()] = "us"
            loc_dict[state.state.lower()] = "us"
        
        # Add missing locations
        loc_dict["uk"] = "gb"
        loc_dict["ny"] = "us"
        loc_dict["nyc"] = "us"
        loc_dict["la"] = "us"
        loc_dict["sf"] = "us"
        loc_dict["bc"] = "ca"
        
        return loc_dict
    

class Data:
    # data container
    
    def __init__(self, train, test):
        self.train = train
        self.test = test
        self.encoder = None
        self.top_accts, self.top_hashtags = self._popular_tags(train)
        self.stemming = Common.stemming_dict()
        self.lemmatization = Common.lemmatization_dict()
        self.stopwords = Common.stopword_list()
        self.loc_dict = Common.create_loc_dict()
                
    @staticmethod
    def _popular_tags(df, num_accts=10, num_hashtags=30):
        """Most mentioned accounts and popular hashtags in training set.

        Args:
            df (DataFrame): dataframe to be worked on.
            num_accts (int): number of most tagged accounts.
            num_hashtags (int): number of most popular hashtags.

        Returns:
            top_accts: list of most tagged accounts.
            top_hashtags: list of most popular hashtags.

        """
        accts = []
        hashtags = []
        for tweet in df.index.values:
            acct = re.findall(r'@\w+', df.loc[tweet, "OriginalTweet"])
            accts.append(acct)
            hashtag = re.findall(r'#\w+', df.loc[tweet, "OriginalTweet"])
            hashtags.append(hashtag)
        
        sort = pd.DataFrame(accts).value_counts(ascending=False)
        top_accts = [tag[0] for tag in sort.index.values[:num_accts]]
        
        sort = pd.DataFrame(hashtags).value_counts(ascending=False)
        # Exclude #covid* and #corova* because those hashtags were used 
        # to collect data, hence they're in most of the tweets.
        top_hashtags = [tag[0] for tag in sort.index.values[:num_hashtags] \
                        if tag[0].lower() != "covid" and 
                        tag[0].lower() != "corona"]
            
        return top_accts, top_hashtags

    @staticmethod
    def _remove_link(text):
        """Remove link from input text.

        Args:
            text (str)

        Returns:
            (str): text without link starting with http.

        """
        return re.sub(r'http\S+', " ", text)
    
    def _remove_accounts(self, text):
        """Remove accounts and only keep most mentioned accounts.

        Args:
            text (str)

        Returns:
            text (str): text without account names except most mentioned 
            accounts.
            
        """
        tags = []   
        
        for tag in self.top_accts:
            tags += re.findall(tag, text)
        text = re.sub(r'@\S+', " ", text)
        for tag in tags:
            text += " " + tag
            
        return text
        
    def _remove_hashtags(self, text):
        """Remove hashtags and only keep most popular hashtags.

        Args:
            text (str)

        Returns:
            text (str): text without hashtags except most popular hashtags.

        """
        tags = []
        
        for tag in self.top_hashtags:
            tags += re.findall(tag, text)
        text = re.sub(r'#\S+', " ", text)
        for tag in tags:
            text += " " + tag
            
        return text
    
    @staticmethod
    def _remove_special_char(text):
        """Remove special characters and numbers.

        Args:
            text (str)

        Returns:
            text (str): text without special characters and numbers.

        """
        special_char = string.punctuation + "0123456789" + "\r" + "\n" + "\t"
        
        for ch in special_char:
            text = text.replace(ch, " ")
            
        return text
    
    @staticmethod
    def _remove_non_english(text):
        """Remove non-English characters.

        Args:
            text:

        Returns:
            text (str): text with all English alphabet.

        """
        clean_text = ""
        
        for word in text.split():
            clean_text += str(np.where(word.isalpha(), (word + " "), ""))
        
        return clean_text
    
    def _remove_stopwords(self, text):
        """Remove stopwords.

        Args:
            text (str)

        Returns:
            (str): text without stopwords.

        """
        return " ".join([word for word in text.split() 
                         if word not in self.stopwords])
    
    def _stemming(self, text):
        """Replace words by its stem.

        Args:
            text (str)

        Returns:
            (str): text with stem of words.

        """
        clean_text = []
        
        for word in text.split():
            if word in self.stemming:
                word = self.stemming[word]
            clean_text.append(word)
            
        return " ".join(clean_text)
    
    def _lemmatization(self, text):
        """Replace words by its lemma.

        Args:
            text (str)

        Returns:
            (str): text with lemma of words.

        """
        clean_text = []
        
        for word in text.split():
            if word in self.lemmatization:
                word = self.lemmatization[word]
            clean_text.append(word)
            
        return " ".join(clean_text)
    
    def _clean_text(self, text):
        """Process text by removing and replacing uninformative words.

        Args:
            text (str)

        Returns:
            text (str): clean text

        """
        text = self._remove_link(text)
        text = self._remove_accounts(text)
        text = self._remove_hashtags(text)
        text = self._remove_special_char(text)
        text = self._remove_non_english(text)
        text = self._remove_stopwords(text)
        text = self._stemming(text)
        text = self._lemmatization(text)
        
        return text
    
    def _clean_location(self, loc):
        """Process location and replace by the country code it's in

        Args:
            loc (str): location

        Returns:
            loc_clean (str): country code

        """
        loc_clean = ""

        if not pd.isnull(loc):
            loc = self._clean_text(loc.lower())
            for sub in loc.split():
                if sub in self.loc_dict:
                    loc_clean = self.loc_dict[sub]
                    break

        return loc_clean
        
    def _clean_tweet(self, df):
        """Clean the Location and OriginalTweet

        Args:
            df (DataFrame): dataframe to be worked on.

        """
        clean_tweet = []
    
        for tweet in df.index.values:
            # clean location
            loc = self._clean_location(df.Location.loc[tweet])
            # clean message
            msg = self._clean_text(df.loc[tweet, 'OriginalTweet'].lower())
            # combine location and tweet to one text
            clean_tweet.append(loc + " " + msg)
        df.loc[:, 'CleanTweet'] = clean_tweet

    def _encode_sentiment(self):
        # Encode and transform Sentiment to numerical values
        self.encoder = LabelEncoder().fit(self.train.Sentiment)
        self.train.Sentiment = self.encoder.transform(self.train.Sentiment)
        self.test.Sentiment = self.encoder.transform(self.test.Sentiment)
        
    def preprocessing(self):
        # Preprocess data
        self._clean_tweet(self.train)
        self._clean_tweet(self.test)
        self._encode_sentiment()

    def inverse_sentiment(self, sentiment):
        """Inverse transform the sentiment from numerical values to string

        Args:
            sentiment (list): list of numerically encoded sentiment

        Returns:
            (list): list of sentiment strings

        """
        return self.encoder.inverse_transform(sentiment)
        

class Tokenizer:
    # Tokenize text
    
    def __init__(self, num_words, oov, maxlen, truncating, padding):
        self.num_words = num_words
        self.oov = oov
        self.maxlen = maxlen
        self.truncating = truncating
        self.padding = padding
        self.tokenizer = None
        self.vocab_size = 0
    
    def set_config(self, **kwargs):
        for (key, value) in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)

    def fit(self, corpus):
        """Fit corpus to tokenizer

        Args:
            corpus (str or list of str)

        """
        self.tokenizer = tf.keras.preprocessing.text.Tokenizer( \
            num_words=self.num_words, oov_token=self.oov)
        self.tokenizer.fit_on_texts(corpus)
        self.vocab_size = len(self.tokenizer.word_index)
        Log.print_msg("INFO", self.vocab_size)
    
    def transform(self, corpus):
        """Transform corpus to tokens.

        Args:
            corpus (str or list of str)

        Returns:
            (list of int): list of tokens padded to same length

        """
        sequences = self.tokenizer.texts_to_sequences(corpus)
        return tf.keras.preprocessing.sequence.pad_sequences(sequences, 
            maxlen=self.maxlen, truncating=self.truncating, 
            padding=self.padding)


class Model:
    # Model container
    
    def __init__(self):
        self.seq_model = None
        self.best_score = {"val_acc": 0, "model": None}
    
    @property
    def model(self):
        # Assign model.
        return self.seq_model.summary()

    @model.setter
    def model(self, model):
        self.seq_model = model
    
    @model.deleter
    def model(self):
        self.seq_model = None
        
    def compile_and_fit(self, train_corpus, train_target, valid_corpus, 
                        valid_target, optimizer='adam', metrics=['accuracy'],
                        loss='sparse_categorical_crossentropy', epochs=4, 
                        batch_size=32):
        """Compile and fit to model.
        
        Args:
            train_corpus: corpus for training
            train_target: sentiment for training
            valid_corpus: corpus for validation
            valid_target: sentiment for validation
            optimizer: optimizer algorithm
            metrics: classification metrics
            loss: loss function
            epochs (int): number of epochs
            batch_size (int): size of batch

        """
        self.seq_model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
        history = self.seq_model.fit(train_corpus, train_target,
                            epochs=epochs, batch_size=batch_size,
                            validation_data=(valid_corpus, valid_target))
        
        self.best_model = history.history['val_accuracy']
    
    @property
    def best_model(self):
        # Record best model.
        return self.best_score
    
    @best_model.setter
    def best_model(self, val_acc):
        if max(val_acc) > self.best_score["val_acc"]:
            self.best_score["val_acc"] = max(val_acc)
            self.best_score["model"] = self.seq_model
    
    @best_model.deleter
    def best_model(self):
        self.best_score = {"val_acc": 0, "model": None}
        
    def evaluate(self, test_corpus, test_target):
            """Predict sentiment of test corpus using best model.
    
            Args:
                test_corpus: corpus for evaluation
                test_target: sentiment for evaluation
    
            Returns:
                (list): loss and accuracy
    
            """
            model = self.best_model['model']
            return model.evaluate(test_corpus, test_target)


if __name__ == "__main__":
    # set log level
    Log.set_log_level("INFO")
    
    # load data
    train_df = pd.read_csv("Corona_NLP_train.csv", encoding='latin1')
    test_df = pd.read_csv("Corona_NLP_test.csv", encoding='latin1')

    data = Data(train_df, test_df)
    data.preprocessing()
    
    num_words=15000
    maxlen=25
    oov="<OOV>"
    trunc='post'
    pad='post'
    embd_dim = 32
    batch = 32
    epoch = 3 
    
    train_target = data.train.Sentiment.values
    test_target = data.test.Sentiment.values
    
    tokenizer = Tokenizer(num_words=num_words, oov=oov, maxlen=maxlen, 
                          truncating=trunc, padding=pad)
    tokenizer.fit(data.train.CleanTweet.values)
    train_corpus = tokenizer.transform(data.train.CleanTweet.values)
    test_corpus = tokenizer.transform(data.test.CleanTweet.values)
    vocab_size = tokenizer.vocab_size
    
    # create Model object and add models
    dnn_model = Model()

    # Conv1D model
    dnn_model.model = tf.keras.models.Sequential([
        tf.keras.layers.Embedding(vocab_size+1, embd_dim, input_length=maxlen),
        tf.keras.layers.Conv1D(embd_dim*8, 5, activation=tf.nn.relu),
        tf.keras.layers.GlobalAveragePooling1D(),
        tf.keras.layers.Dense(128, activation=tf.nn.relu),
        tf.keras.layers.Dense(5, activation=tf.nn.sigmoid)
    ])
    print(dnn_model.model)
    dnn_model.compile_and_fit(train_corpus, train_target, test_corpus, \
                              test_target, epochs=epoch, batch_size=batch)

    # Bidirectional LSTM
    dnn_model.model = tf.keras.models.Sequential([
        tf.keras.layers.Embedding(vocab_size+1, embd_dim, input_length=maxlen),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(embd_dim*8)),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(128, activation=tf.nn.relu),
        tf.keras.layers.Dense(5, activation=tf.nn.sigmoid)
    ])
    print(dnn_model.model)
    dnn_model.compile_and_fit(train_corpus, train_target, test_corpus, \
                              test_target, epochs=epoch, batch_size=batch)

    # Bidirectional GRU
    dnn_model.model = tf.keras.models.Sequential([
        tf.keras.layers.Embedding(vocab_size+1, embd_dim, input_length=maxlen),
        tf.keras.layers.Bidirectional(tf.keras.layers.GRU(embd_dim*8)),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(128, activation=tf.nn.relu),
        tf.keras.layers.Dense(5, activation=tf.nn.sigmoid)
    ])
    print(dnn_model.model)
    dnn_model.compile_and_fit(train_corpus, train_target, test_corpus, \
                              test_target, epochs=epoch, batch_size=batch)
    
    # Show best model and val_accuracy
    print("Best model is", dnn_model.best_model)
    
    # Evaluate model with test corpus and show accuracy
    test_pred = dnn_model.evaluate(test_corpus, test_target)
    print("Test corpus accuracy is", test_pred[1])