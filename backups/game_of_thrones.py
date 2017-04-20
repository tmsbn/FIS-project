import pandas as pd

from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import tree
import numpy as np


def main():

    cols_to_drop = ['is_alive', 'name', 's_no']

    no_of_outputs = 2

    # Read csv and make data frame
    df = pd.read_csv('character_predictions.csv')

    # convert categorical variable
    df_dummies = pd.get_dummies(df, columns=['house'])


    # Extract the test row
    df_test = df_dummies.tail(2).drop(cols_to_drop, axis=1)


    # Remove outputs
    df_input = df_dummies[:-no_of_outputs]

    # Select Target values
    df_target_values = df_input['is_alive'].values

    # Select Input values
    df_input_values = df_input.drop(cols_to_drop, axis=1).iloc[:, :].values

    cw = {
        2: 4
    }

    # Build classifier
    clf = DecisionTreeClassifier()
    clf = clf.fit(df_input_values, df_target_values)


    # # Models
    # decisionTreeClassifer = DecisionTreeClassifier()
    # RandomFor
    #
    # models = [), ]
    #
    # predicted_value = clf.predict(df_test.values)
    # predicted_prob = clf.predict_proba(df_test.values)
    # print(predicted_value)
    # print(predicted_prob)


if __name__ == '__main__':
    main()
