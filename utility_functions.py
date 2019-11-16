#==========================================
# Title: Utility functions
# Author: Rajesh Gupta
# Date:   16 Nov 2019
#==========================================
from datetime import datetime
import logging
import json
import os
import pickle as pkl

# Configuring the logger to record logs across the project
def configure_logger():
    current_datetime = datetime.now().strftime("%d-%m-%Y %H-%M-%S")
    fpath = os.path.join(os.getcwd(), os.path.join("logs", "multi-label-classification-log-{}.txt".format(current_datetime)))
    log_file = fpath
    log_stream = 'sys.stdout'
    log_level = logging.DEBUG
    log_format = "%(asctime)s %(levelname)s: %(message)s"
    log_date_format = "%m/%d/%Y %H:%M:%S"
    logging.basicConfig(filename=log_file, level=log_level, format=log_format, datefmt=log_date_format)

# Simple function to write to a text file
def write_file(text, filename):
    with open(filename,"w") as f:
        f.write(text)

# Simple function read a text file
def read_file(filename):
    with open(filename, "r") as f:
        data = f.read()
    return data

# Standard function to pickle .py objects
def pickle(obj, filename, foldername):
    PICKLE_PATH = os.path.join(os.getcwd(), foldername, filename + ".pkl") 
    with open(PICKLE_PATH, "wb") as f:
        pkl.dump(obj, f)

# Standard function to unpickle files to .py objects
def unpickle(filename, foldername):
    UNPICKLE_PATH = os.path.join(os.getcwd(), foldername, filename + ".pkl")
    with open(UNPICKLE_PATH, "rb") as f:
        data = pkl.load(f)
    return data

# Writing to a JSON file
def write_JSON(obj, filename, foldername):
    WRITE_PATH = os.path.join(os.getcwd(), foldername, filename + ".json") 
    with open(WRITE_PATH, "w") as f:
        json.dump(obj, f)
        
# Reading from a JSON file
def read_JSON(filename, foldername):
    READ_PATH = os.path.join(os.getcwd(), foldername, filename + ".json") 
    with open(READ_PATH, "r") as f:
        data = json.load(f)
    return data