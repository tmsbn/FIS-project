import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import tree
import numpy as np


def main():
    data_frame = pd.read_csv('test_data_set.csv')
    df_with_dummies = pd.get_dummies(data_frame, columns=['house'])
    print(df_with_dummies)

if __name__ == '__main__':
    main()