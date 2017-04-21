import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
from sklearn.tree import _tree
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
import pydot


def tree_to_code(d_tree, feature_names):
    tree_ = tree.tree_
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
        ]
    print("def tree({}):".format(", ".join(feature_names)))

    def recurse(node, depth):
        indent = "  " * depth
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]
            threshold = tree_.threshold[node]
            print("{}if {} <= {}:".format(indent, name, threshold))
            recurse(tree_.children_left[node], depth + 1)
            print("{}else:  # if {} > {}".format(indent, name, threshold))
            recurse(tree_.children_right[node], depth + 1)
        else:
            print("{}return {}".format(indent, tree_.value[node]))

    recurse(0, 1)


def visualize_tree(d_tree, feature_names):
    with open("output.dot", 'w') as f:
        export_graphviz(d_tree, out_file=f, feature_names=feature_names)

        try:
            (graph,) = pydot.graph_from_dot_file('output.dot')
            graph.write_png('output.png')
        except Exception as e:
            print(str(e))


def main():
    target_col = 'label'

    # Read csv and make data frame
    df = pd.read_csv('voice.csv')
    #  print(df)

    # Get Column names
    column_names = df.columns.values

    # Test and Training
    df_train, df_test = train_test_split(df, test_size=0.3)

    # Select Input values
    df_input_values = df_train.iloc[:, :-1].values

    # Select Target values
    df_target_values = df_train[target_col].values

    actual_values = df_test[target_col].values
    df_test_values = df_test.iloc[:, :-1].values

    dt_clf = DecisionTreeClassifier()
    rf_clf = RandomForestClassifier()
    lg_clf = LogisticRegression()

    models = [lg_clf, dt_clf, rf_clf]

    for j in range(0, 5):

        print("Iteration " + str(j + 1) + ":\n")

        for model in models:

            clf = model.fit(df_input_values, df_target_values)

            # Predict values
            predicted_value = clf.predict(df_test_values)
            count = 0
            value_length = len(actual_values)
            print(value_length)

            confusion_matrix = [0] * 4

            for i in range(0, value_length):

                if predicted_value[i] == actual_values[i]:
                    count += 1

                if predicted_value[i] == actual_values[i]:
                    if predicted_value[i] == 1:
                        confusion_matrix[3] += 1
                    else:
                        confusion_matrix[0] += 1
                if predicted_value[i] != actual_values[i]:
                    if predicted_value[i] == 1:
                        confusion_matrix[1] += 1
                    else:
                        confusion_matrix[2] += 1

            print(type(clf).__name__)
            print(str(count) + ' out of ' + str(value_length) + ' which is ' + str(count / value_length * 100.0))
            print('Confusion matrix:')
            print(confusion_matrix)

            confusion_matrix.clear()

        print("\n")

        # visualize_tree(clf, column_names)


if __name__ == "__main__":
    main()
