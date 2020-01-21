#==========================================
# Title: Cleaning scripts
# Author: Rajesh Gupta
# Date:   16 Nov 2019
#==========================================
from utility_functions import *
from config_vars import *
import re, pandas as pd, numpy as np
import logging, string
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer

class DataCleaning:

    stop = stopwords.words('english')
    ps = PorterStemmer()
    wnl = WordNetLemmatizer()

    def __init__(self):
        self.js = re.compile(r'<script.*?>.*?</script>')
        self.css = re.compile(r'<style.*?>.*?</style>')
        self.html = re.compile(r'<.*?>')
        self.braces = re.compile(r'{.*?}')
        self.spl_symbols = re.compile(r'&\S*?;|[\\\/\_\(\)\|\>\<\%]|\.async\-hide|wikipedia|free|encyclopedia|403|Forbidden|nginx', re.I)
        self.spaces = re.compile(r'\s+')
        self.urls = re.compile(r'https?\:\/\/.*?\s')
        self.non_alpha = re.compile(r'[^a-zA-Z\s]')
        self.extra_spaces = re.compile(r'\s+')

    def clean_metadata(self, df, columns=None):
        logging.info("="*15+"Cleaning metadata"+"="*15)
        if columns is not None:
            df = df[columns].to_frame().applymap(lambda x: self.clean_metadata(x))
            return df.T.iloc[0,:] # df.squeeze() will also do
        return self.meta_data.sub(" ", str(df))

    def clean_extra_spaces(self, df, columns=None):
        logging.info("="*15+"Cleaning extra whitespaces"+"="*15)
        if columns is not None:
            df = df[columns].to_frame().applymap(lambda x: self.clean_extra_spaces(x))
            return df.T.iloc[0,:] # df.squeeze() will also do
        return self.extra_spaces.sub(" ", str(df)).strip()

    def clean_js(self, df, columns=None):
        """
        Aliter : PLEASE NOTE THAT THIS CAN BE AN ALITER TO ALL FUNCTIONS IN THIS CLASS.
        if columns is not None:
            y=[]
            for x in columns:
                y.append(self.js.sub(" ", str(df[x])))
            return pd.Series(y)
        """
        logging.info("="*15+"Cleaning JS"+"="*15)
        if columns is not None:
            df = df[columns].to_frame().applymap(lambda x: self.clean_js(x))
            return df.T.iloc[0,:] # df.squeeze() will also do
        return self.js.sub(" ", str(df))

    def clean_non_alpha(self, df, columns=None):
        logging.info("="*15+"Cleaning Non-alphabet characters"+"="*15)
        if columns is not None:
            df = df[columns].to_frame().applymap(lambda x: self.clean_non_alpha(x))
            return df.T.iloc[0,:] # df.squeeze() will also do
        return self.non_alpha.sub("", str(df))

    def clean_html(self, df, columns=None):
        logging.info("="*15+"Cleaning HTML"+"="*15)
        if columns is not None:
            df = df[columns].to_frame().applymap(lambda x: self.clean_html(x))
            return df.T.iloc[0,:] # df.squeeze() will also do
        return self.html.sub(" ", str(df))

    def clean_css(self, df, columns=None):
        logging.info("="*15+"Cleaning CSS"+"="*15)
        if columns is not None:
            df = df[columns].to_frame().applymap(lambda x: self.clean_css(x))
            return df.T.iloc[0,:] # df.squeeze() will also do
        return self.css.sub(" ", str(df))

    def clean_braces(self, df, columns=None):
        logging.info("="*15+"Cleaning braces"+"="*15)
        if columns is not None:
            df = df[columns].to_frame().applymap(lambda x: self.clean_braces(x))
            return df.T.iloc[0,:] # df.squeeze() will also do
        return self.braces.sub(" ", str(df))

    def clean_special_symbols(self, df, columns=None):
        logging.info("="*15+"Cleaning special symbols"+"="*15)
        if columns is not None:
            df = df[columns].to_frame().applymap(lambda x: self.clean_special_symbols(x))
            return df.T.iloc[0,:] # df.squeeze() will also do
        return self.spl_symbols.sub(" ", str(df))

    def clean_spaces(self, df, columns=None):
        logging.info("="*15+"Cleaning spaces"+"="*15)
        if columns is not None:
            df = df[columns].to_frame().applymap(lambda x: self.clean_spaces(x))
            return df.T.iloc[0,:] # df.squeeze() will also do
        return self.spaces.sub(" ", str(df))

    def clean_urls(self, df, columns=None):
        logging.info("="*15+"Cleaning URLs"+"="*15)
        if columns is not None:
            df = df[columns].to_frame().applymap(lambda x: self.clean_urls(x))
            return df.T.iloc[0,:] # df.squeeze() will also do
        return self.urls.sub(" ", str(df))

    @classmethod
    def clean_stopwords(cls, df, columns=None):
        logging.info("="*15+"Cleaning stopwords"+"="*15)
        if columns is not None:
            df = df[columns].to_frame().applymap(lambda x: cls.clean_stopwords(x))
            return df.T.iloc[0,:] # df.squeeze() will also do
        return " ".join([token for token in str(df).split() if token not in cls.stop])
    
    @classmethod
    def stem_tokens(cls, df, columns=None):
        logging.info("="*15+"Stemming words"+"="*15)
        if columns is not None:
            df = df[columns].to_frame().applymap(lambda x: cls.stem_tokens(x))
            return df.T.iloc[0,:] # df.squeeze() will also do
        return " ".join([cls.ps.stem(token) for token in str(df).split()])

    @classmethod
    def lemmatize_tokens(cls, df, columns=None):
        logging.info("="*15+"Lemmatizing words"+"="*15)
        if columns is not None:
            df = df[columns].to_frame().applymap(lambda x: cls.lemmatize_tokens(x))
            return df.T.iloc[0,:] # df.squeeze() will also do
        return " ".join([cls.wnl.lemmatize(token) for token in str(df).split()])

    @classmethod
    def preprocessing(cls, data):
        try:
            # Remove punctuations
            data = [str(_char) for _char in data if _char not in string.punctuation]
            # Changing back to text
            data = "".join(data)
        except Exception:
            data = str(0)
        return data