import pandas as pd

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import tree
import numpy as np


def main():

    no_of_outputs = 2

    # Read csv and make data frame
    df = pd.read_csv('character_predictions.csv')
    print(df)

    # convert categorical variable
    df_dummies = pd.get_dummies(df, columns=['house'])

    # Extract the predict row
    df_predict = df_dummies.tail(2).drop(['is_alive', 'name', 's_no'], axis=1)

    # Remove outputs
    df_input = df_dummies[:-no_of_outputs]

    # Select Target values
    df_target_values = df_input['is_alive'].values

    # Select Input values
    df_input_values = df_input.drop(['is_alive', 'name', 's_no'], axis=1).iloc[:, :].values

    cw = {
        2: 4
    }

    # Build classifier
    clf = DecisionTreeClassifier(class_weight=cw)
    clf = clf.fit(df_input_values, df_target_values)

    predicted_value = clf.predict(df_predict)
    print(predicted_value)


if __name__ == '__main__':
    main()
