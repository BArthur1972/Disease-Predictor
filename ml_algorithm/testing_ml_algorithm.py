import pandas as pd
from scipy.stats import mode
from ml_algorithm import (ensemble,
                        metrics,
                        naive_bayes,
                        neighbors,
                        skpre,
                        svm,
                        tree)

def main():
    # Reading the testing dataset.
    test_data = pd.read_csv("Testing.csv").dropna(axis=1)


    encoder = skpre.LabelEncoder()

    # Splitting the data into features(test_X) and target(test_Y)
    test_X = test_data.iloc[:, :-1]
    test_Y = test_data.iloc[:, -1]

    # Encoding the target column into numbers
    test_data["prognosis"] = encoder.fit_transform(test_data["prognosis"])

    # Storing the models in variables.
    final_knn_model = neighbors.KNeighborsClassifier()
    final_nb_model = naive_bayes.GaussianNB()
    final_dtc_model = tree.DecisionTreeClassifier()
    final_rf_model = ensemble.RandomForestClassifier(random_state=21)
    final_svm_model = svm.SVC()

    # Fitting the models on the features and target
    final_nb_model.fit(test_X, test_Y)
    final_rf_model.fit(test_X,test_Y)
    final_dtc_model.fit(test_X,test_Y)
    final_knn_model.fit(test_X,test_Y)
    final_svm_model.fit(test_X,test_Y)

    nb_preds = final_nb_model.predict(test_X)
    rf_preds = final_rf_model.predict(test_X)
    svm_preds = final_svm_model.predict(test_X)
    dtc_preds = final_dtc_model.predict(test_X)
    knn_preds = final_knn_model.predict(test_X)

    # Taking the mode of all all the predictions from each model.
    final_preds = [mode([i,j,k,l,m])[0][0] for i,j,k,l,m in zip(nb_preds, rf_preds, svm_preds, dtc_preds, knn_preds)]

    # Printing the performance metriics of the models
    print(f"Accuracy on Test dataset by the combined model: {metrics.accuracy_score(test_Y, final_preds)}")

    print(f"Sensivity on Test dataset by the combined model: {metrics.recall_score(test_Y, final_preds, average = 'macro')}")

    conf_matrix = metrics.confusion_matrix(test_Y, final_preds)
    print(f"Confusion matrix for the combined model: {conf_matrix}")

if __name__ == "__main__":
    main()