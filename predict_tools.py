import numpy as np
import joblib
from scipy.stats import mode
from fuzzywuzzy import fuzz
from directory import SYMPTOM_NAMES, DATA_DICT

# Takes in input from the user and stores them in a list separated by commas.
def take_input(input_text):
    symptoms = input_text.split(",")

    return symptoms

# Create a list containing symptoms in a form that can be recognized by models.
def create_new_symptoms(symptoms):
    new_symptoms = []

    for symptom in symptoms:
        for name in SYMPTOM_NAMES.keys():
            score1 = fuzz.token_sort_ratio(symptom, name)
            score2 = fuzz.partial_ratio(symptom, name)
            if score1 >= 60 or score2 >= 70:
                new_symptoms.append(name)

    return new_symptoms

# Creating input data for the models
def create_input_data(new_symptoms):
    input_data = [0] * len(DATA_DICT["symptom_index"])

    for symptom in new_symptoms:
        index = DATA_DICT["symptom_index"][symptom]
        input_data[index] = 1

    input_data = np.array(input_data).reshape(1,-1)

    return input_data

# Loads the machine learning models and uses them to predict a disease based on the input data.
def predict_disease(input_data):
    svm_from_joblib = joblib.load('svm model.pkl')
    knn_from_joblib = joblib.load('k_nearest_neighbors model.pkl')
    nb_from_joblib = joblib.load('naive_bayes_model.pkl')
    rf_from_joblib = joblib.load('random_forest_model.pkl')
    dtc_from_joblib = joblib.load('decision_tree_model.pkl')

    rf_prediction = DATA_DICT["predictions_classes"][rf_from_joblib.predict(input_data)[0]]
    nb_prediction = DATA_DICT["predictions_classes"][nb_from_joblib.predict(input_data)[0]]
    svm_prediction = DATA_DICT["predictions_classes"][svm_from_joblib.predict(input_data)[0]]
    knn_prediction = DATA_DICT["predictions_classes"][knn_from_joblib.predict(input_data)[0]]
    dtc_prediction = DATA_DICT["predictions_classes"][dtc_from_joblib.predict(input_data)[0]]

    final_prediction = mode([rf_prediction, nb_prediction, svm_prediction, knn_prediction, dtc_prediction])[0][0]

    return final_prediction
