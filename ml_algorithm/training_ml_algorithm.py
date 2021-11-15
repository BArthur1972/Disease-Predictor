from sklearn import (ensemble,
                     metrics,
                     model_selection as skms,
                     naive_bayes,
                     neighbors,
                     preprocessing as skpre,
                     svm,
                     tree)
import pandas as pd
import joblib

def main():
    dataset = pd.read_csv("Training.csv")

    # identify null values
    null_columns = dataset.columns[dataset.isnull().any()]
    print(dataset[null_columns].isnull().sum())

    # dropping 'Unnamed: 133'
    dataset = dataset.dropna(axis=1)

    # Encoding the target value into numerical
    # value using LabelEncoder
    encoder = skpre.LabelEncoder()
    dataset["prognosis"] = encoder.fit_transform(dataset["prognosis"])

    ## Spliting the target from the features.
    ftr = dataset.iloc[:,:-1]
    tgt = dataset.iloc[:, -1]

    ## Train Test Split
    (train_ftr, test_ftr, 
    train_tgt, test_tgt) = skms.train_test_split(ftr, tgt, test_size = 0.25, random_state = 21)

    print(f"Train: {train_ftr.shape}, {train_tgt.shape}")
    print(f"Test: {test_ftr.shape}, {test_tgt.shape}")

    # Creating a dictionary to store classifiers
    models = {
        "Decision Tree Classifier" : tree.DecisionTreeClassifier(),
        "KNN" : neighbors.KNeighborsClassifier(),
        "Naive Bayes" : naive_bayes.GaussianNB(),
        "SVC" : svm.SVC(),
        "Random Forest" : ensemble.RandomForestClassifier(random_state = 21)
    }

    # Printing metrics to show performance for each classifier
    for model_name, model in models.items():
        tgt_preds = (model.fit(train_ftr, train_tgt).predict(test_ftr))
        print(model_name)
        print("="*len(model_name))
        print("accuracy:",metrics.accuracy_score(test_tgt, tgt_preds))
        print("precision:",metrics.precision_score(test_tgt, tgt_preds, average = 'weighted'))
        print("sensitivity:", metrics.recall_score(test_tgt, tgt_preds, average = 'weighted'))
        print("confusion matrix:", metrics.confusion_matrix(test_tgt, tgt_preds))
        print("\n")

    # Storing trained models in variables.
    final_svm_model = svm.SVC().fit(ftr.values, tgt)
    final_nb_model = naive_bayes.GaussianNB().fit(ftr.values, tgt)
    final_rf_model = ensemble.RandomForestClassifier(random_state=21).fit(ftr.values, tgt)
    final_knn_model = neighbors.KNeighborsClassifier().fit(ftr.values, tgt)
    final_dtc_model = tree.DecisionTreeClassifier().fit(ftr.values, tgt)

    # Saving the model as a pickle in a file using joblib
    joblib.dump(final_svm_model, 'svm model.pkl')
    joblib.dump(final_nb_model, 'naive_bayes_model.pkl')
    joblib.dump(final_rf_model, 'random_forest_model.pkl')
    joblib.dump(final_knn_model, 'k_nearest_neighbors model.pkl')
    joblib.dump(final_dtc_model, 'decision_tree_model.pkl')

if __name__ == "__main__":
    main()