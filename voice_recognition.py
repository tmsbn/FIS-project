import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


def main():
    # Read csv and make data frame
    df = pd.read_csv('voice.csv')

    # Test and Training
    df_train, df_test = train_test_split(df, test_size=0.3)

    # Select Input values
    df_input_values = df_train.iloc[:, :-1].values

    # Select Target values
    df_target_values = df_train['label'].values

    actual_values = df_test['label'].values
    df_test_values = df_test.iloc[:, :-1].values

    # clf = DecisionTreeClassifier()
    clf = RandomForestClassifier()
    clf = clf.fit(df_input_values, df_target_values)

    # Predict values
    predicted_value = clf.predict(df_test_values)

    count = 0
    al = len(actual_values)
    for i in range(0, al):
        if predicted_value[i] == actual_values[i]:
            count += 1
        # print(predicted_value[i], actual_values[i])

    print("count is:" + str(count) + ' out of ' + str(al) + ' which is ' + str(count/al * 100.0))

if __name__ == "__main__":
    main()
