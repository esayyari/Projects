import numpy as np
from sklearn.datasets import load_svmlight_file

def read_data(data_file_name):
    X, y = load_svmlight_file(data_file_name)
    return (X,y)
