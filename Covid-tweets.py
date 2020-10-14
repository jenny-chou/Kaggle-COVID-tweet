# -*- coding: utf-8 -*-
"""
Created on Tue Oct 13 14:55:36 2020

@author: jenny
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
        """
        Set permissible log level.

        Args:
            level (str, optional): level to be set. Defaults to "INFO".

        Returns:
            None.
            
        """
        cls.LEVEL = cls.LOG_LEVEL[level]
    
    @classmethod
    def print_msg(cls, level, msg):
        """
        Check log level and print permissible logs.

        Args:
            level (str): level of the message.
            msg (str): message to be print.

        Returns:
            None.
            
        """
        if cls.LOG_LEVEL[level] >= cls.LEVEL:
            print(msg)        

class Common:
    lemmatization_dict = {"shop":"store", "supermarket":"market", 
        "much":"many", "employee":"worker", "staff":"worker", 
        "global":"world", "company":"business", "consumer":"customer", 
        "house":"home", "grocery":"goods", "products":"goods", 
        "toilet":"toiletpaper", "paper":"toiletpaper"}
    
    stemming_dict = {"buying":"buy", "bought":"buy", "working":"work", 
        "worked":"work", "shopping":"shop", "shopped":"shop",
        "shops":"shop", "days":"day", "weeks":"week", "masks":"mask", 
        "got":"get", "gets":"get", "consumers":"consumer", "gets":"get", 
        "supermarkets":"supermarket", "says":"say", "saying":"say", 
        "said":"say", "getting":"get", "companies":"company", "went":"go", 
        "got":"get", "making":"make", "made":"make", "services":"service", 
        "hours":"hour", "years":"year", "products":"product", "going":"go", 
        "increases":"increase", "increased":"increase", "markets":"market", 
        "close":"closed", "needs":"need", "customers":"customer", 
        "needed":"need", "hands":"hand", "stores":"store", "staffs":"staff", 
        "employees":"employee", "workers":"worker", "businesses":"business",}

class Data:
    # data container
    
    def __init__(self, train, test):
        self.train = train
        self.test = test

        
    def word_locations():
        # Below files are used for data cleaning - Retrieving countries with help of city codes/state codes...(optional)
        states = json.loads(requests.get("https://raw.githubusercontent.com"\
            "/praneshsaminathan/country-state-city/master/states.json").text)
        countries=json.loads(requests.get("https://raw.githubusercontent.com"\
            "/praneshsaminathan/country-state-city/master/"\
            "countries.json").text)
        cities=json.loads(requests.get("https://raw.githubusercontent.com/"\
            "praneshsaminathan/country-state-city/master/cities.json").text)
        us_states_code=pd.read_csv("https://worldpopulationreview.com/"\
            "static/states/abbr-name.csv",names=['state_code','state'])
        
        states = states["states"]
        countries = countries["countries"]
        cities = cities["cities"]

        world_dict = {}
    
        for city in cities:
            state_id = int(re.findall(r'\d+', city["state_id"])[0])
            country_id = states[state_id-1]["country_id"]
            country_id = int(re.findall(r'\d+', country_id)[0])
            country_name = countries[country_id-1]["sortname"]
            world_dict[city["name"].lower()] = country_name.lower()
            
        for state in states:
            country_id = int(re.findall(r'\d+', state["country_id"])[0])
            country_name = countries[country_id-1]["sortname"]
            world_dict[state["name"].lower()] = country_name.lower()
            
        for country in countries:    
            world_dict[country["sortname"].lower()] = country["sortname"].lower()
            world_dict[country["name"].lower()] = country["sortname"].lower()
            
        for index in us_states_code.index:
            state = us_states_code.loc[index]
            world_dict[state.state_code.lower()] = "us"
            world_dict[state.state.lower()] = "us"
            
        world_dict["uk"] = "gb"
        world_dict["ny"] = "us"
        world_dict["nyc"] = "us"
        world_dict["la"] = "us"
        world_dict["sf"] = "us"
        world_dict["bc"] = "ca"
        
        return world_dict

    def remove_link(sentence):
        return re.sub(r'http\S+', " ", sentence)
    
    def remove_tag(sentence):
        keep_tag = []
        for tag in top_tags:
            keep_tag += re.findall(tag, sentence)        
        sentence = re.sub(r'@\S+', " ", sentence)    
        for tag in keep_tag:
            sentence += " " + tag        
        return sentence
        
    def remove_hashtag(sentence):
        keep_tag = []
        for tag in top_hashtags:
            keep_tag += re.findall(tag, sentence)        
        sentence = re.sub(r'#\S+', " ", sentence)    
        for tag in keep_tag:
            sentence += " " + tag        
        return sentence
        
    def remove_special_char(sentence):
        for special_ch in (string.punctuation + "0123456789" + "\r" + "\n"):
            sentence = sentence.replace(special_ch, " ")
        return sentence
    
    def remove_non_english(sentence):
        clean_sentence = ""
        for word in sentence.split():
            clean_sentence += str(np.where(word.isalpha(), (word + " "), ""))
        return clean_sentence
    
    def remove_stop_words(sentence):
        for word in ["covid", "corona", "virus"]:
            sentence = re.sub(word, " ", sentence)
        sentence = [word for word in sentence.split() if word not in stop_words]
        return " ".join(sentence)
    
    def stemming_and_lemmatization(sentence):
        clean_sentence = []
        for word in sentence.split():
            if word in stemming_dict:
                word = stemming_dict[word]
            if word in lemmatization_dict:
                word = lemmatization_dict[word]
            clean_sentence.append(word)    
        return " ".join(clean_sentence)
    
    def clean_sentence(sentence):
        sentence = remove_link(sentence)
        sentence = remove_tag(sentence)
        sentence = remove_hashtag(sentence)
        sentence = remove_special_char(sentence)
        sentence = remove_non_english(sentence)
        sentence = remove_stop_words(sentence)
        sentence = stemming_and_lemmatization(sentence)
        return sentence





if __name__ == "__main__":
    # set log level
    Log.set_log_level("INFO")
    
    # load data
    train_df = pd.read_csv("Corona_NLP_train.csv", encoding='latin1')
    test_df = pd.read_csv("Corona_NLP_test.csv", encoding='latin1')
    
    # process training set and validation set
    train_features, train_target, valid_features, valid_target = \
        process_data(train, validation, True)
    
    # create Model object and add models