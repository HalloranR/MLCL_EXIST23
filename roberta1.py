#imports
import sys
import pandas as pd
import numpy as np
import transformers
from transformers import RobertaModel, RobertaTokenizer

if __name__ == '__main__':
    if(len(sys.argv) < 1):
        raise Exception("error")
    
    training_str = sys.argv[1]

    training_df = pd.read_json(training_str).T
    print(training_df.head())