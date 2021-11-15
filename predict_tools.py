import numpy as np
import joblib
from scipy.stats import mode
from fuzzywuzzy import fuzz
from directory import SYMPTOM_NAMES, DATA_DICT


def take_input(input_text):
    '''This function takes in input_text and returns a list containing the symptoms in the string found in input_text seperated by commas.

    Args:
        input_text: The syptoms that the user enters

    Returns:
        new_symptoms: A list containing the symptoms in the string found in input_text seperated by commas.
    '''
    symptoms = input_text.split(",")

    return symptoms


def create_new_symptoms(symptoms):
    '''This function creates and returns a list containing the user's symptoms after matching them with the symptom names in SYMPTOM_NAMES to pick the closest matches.

    Args:
        symptoms: a list containing the users symptoms. 

    Returns:
        new_symptoms: A list containing the closest matches to the user's symptoms found in SYPTOM_NAMES. 
    '''
    new_symptoms = []

    for symptom in symptoms:
        for name in SYMPTOM_NAMES.keys():
            token_sort_score = fuzz.token_sort_ratio(symptom, name)
            token_set_score = fuzz.token_set_ratio(symptom, name)
            partial_ratio_score = fuzz.partial_ratio(symptom, name)
            if token_sort_score >= 60 or token_set_score >= 70 or partial_ratio_score >= 70:
                new_symptoms.append(name)

    return new_symptoms


def create_input_data(new_symptoms):
    '''This function creates a numpy array containing 0's and 1's representing symptoms. O meaning the user doesn't have that symptom and 1 meaning the user has it.

    Args:
        new_symptoms: a list containing the user's symptoms.

    Returns:
        input_data: a numpy array containing 0's and 1's representing the symptoms that the user has and does not have respectfully.
    '''
    input_data = [0] * len(DATA_DICT["symptom_index"])

    for symptom in new_symptoms:
        index = DATA_DICT["symptom_index"][symptom]
        input_data[index] = 1

    input_data = np.array(input_data).reshape(1, -1)

    return input_data


def predict_disease(input_data):
    '''This function takes input_data and loads the trained machine learning models using the joblib library. It then uses these models to predict the user's disease based on the data in the numpy array. It does this by taking the mode of the all five predictions from all five models. 

    Args:
        input_data:  a numpy array containing 0's and 1's representing the symptoms that the user has and does not have respectfully.

    Returns:
        final_prediction: the user's predicted disease.
    '''
    svm_from_joblib = joblib.load('saved_models/svm model.pkl')
    knn_from_joblib = joblib.load('saved_models/k_nearest_neighbors model.pkl')
    nb_from_joblib = joblib.load('saved_models/naive_bayes_model.pkl')
    rf_from_joblib = joblib.load('saved_models/random_forest_model.pkl')
    dtc_from_joblib = joblib.load('saved_models/decision_tree_model.pkl')

    rf_prediction = DATA_DICT["predictions_classes"][rf_from_joblib.predict(
        input_data)[0]]
    nb_prediction = DATA_DICT["predictions_classes"][nb_from_joblib.predict(
        input_data)[0]]
    svm_prediction = DATA_DICT["predictions_classes"][svm_from_joblib.predict(
        input_data)[0]]
    knn_prediction = DATA_DICT["predictions_classes"][knn_from_joblib.predict(
        input_data)[0]]
    dtc_prediction = DATA_DICT["predictions_classes"][dtc_from_joblib.predict(
        input_data)[0]]

    final_prediction = mode([
        rf_prediction, nb_prediction, svm_prediction, knn_prediction,
        dtc_prediction
    ])[0][0]

    return final_prediction
