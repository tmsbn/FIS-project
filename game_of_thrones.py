import pandas as pd

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import tree
import numpy as np


def main():
    data_frame = pd.read_csv('test_data_set.csv')
    print(data_frame)

    # dummies
    df_with_dummies = pd.get_dummies(data_frame, columns=['house'])

    clf = tree.DecisionTreeClassifier()
    target_column = df_with_dummies['is_alive'].values
    df_data = df_with_dummies.drop('is_alive', axis=1)
    print(df_data.loc[:, :].values)


if __name__ == '__main__':
    main()
