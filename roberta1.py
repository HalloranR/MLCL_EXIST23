#imports
import sys
import pandas as pd
import numpy as np
import tensorflow as tf
import transformers
from transformers import RobertaModel, RobertaConfig, RobertaTokenizer

if __name__ == '__main__':
    if(len(sys.argv) < 2):
        raise Exception("error")
    
    #import the datasets as strings
    training_str = sys.argv[1]
    testing_str = sys.argv[2]

    training_df = pd.read_json(training_str).T
    testing_df = pd.read_json(training_str).T

    configuration = RobertaConfig()
    model = RobertaModel(configuration)

    print(testing_df.head())