# Disease Predictor

## Introduction
This is a command-line-based interactive program that will ask the user to enter their symptoms. I have trained five machine learning models on a dataset that includes a set of symptoms and their corresponding diseases.

When the program is initialized, it will ask the user how he or she is feeling and ask them to enter their symptoms. The program will store these symptoms as a numpy array and implement the machine learning algorithms on them. It will return the predicted disease which will be displayed as output.

## How to Run
- To run the program, first download all the files and upload them to an IDE of your choice.
- Next, you need to go to the terminal or shell and run the following code snippet:
```python
python app.py
```
- The program should ask you to enter your symptoms separated by commas like so:
```Dehydration,Loss Of Appetite, Abdominal Pain, Diarrhoea```

- Enter the symptoms and hit enter to get your diagnosis.

## File Description
### 1. app.py
- This is where the driver code for the program is stored and executed from. 
- It contains the functions:

i. ```create_gui()``` - which sets up all the widgets in the Tkinter GUI.

ii. ```on_predict()``` - which makes use of all the functions in predict_tools.py to predict the user's disease.

iii. ```clear_textbox()```- which clears the textbox when the clear button is pressed.

iv. ```main()```- which executes the program.

### 2. predict_tools.py

In this file you will find all the helper fuctions for this program. They are listed below:    
i. ```take_input()```
    - Takes in input from the user and stores them in a list separated by commas.

ii. ```create_input_symptoms()```
    - Creates a list containing symptoms in a form that can be recognized by models using fuzzy string matching.

iii. ```create_input_data()```
    - Creates a numpy array containing entries such that the users inputs assigned to 1 and the remaining assigned to 0.

iv. ```predict_disease()```
    - Loads the machine learning models and uses them to predict a disease based on the input data. It does this by taking the mode(most occurrences) of all five predictions since there are five models trained with five different classifiers.

### 3. ml_algorithm.py
- In this file you will find the code which I used to build, train and evaluate all five machine learning models. You can run this file to see how well the models are performing. I used the following metrics to judge.
    * precision
    * accuracy
    * sensitivity
    * confusion matrix

- To run this file, go to the terminal or shell and run the following code snippet:
``` python
 python ml_algorithm.py
```
- You will see the evaluation metrics for all five models printed.

### 4. test_ml_algorithm.py
- In this file, I will implement the trained models on Testing.csv. This csv file is similar to Training.csv but is smaller and has different entries. If I get similar results for the metrics accounted for while training the models, then I can say that the models are working well.
- To run the file, run the code snippet below in the terminal or shell:
```python
python test_ml_algorithm.py
```
- You should see the performance metrics of the models on the testing dataset. 

### 4. directory.py
- This file contains the data structures that hold the valid symptom names and disease names. They are:
    * SYMPTOM_NAMES - which is a dictionary containing all the valid symptom names as keys and their corresponding indices as values.
    * ORIGINAL_PROGNOSIS - a numpy array containing all the valid prognosis names.
    * DATA_DICT - a dictionary containing symptom_indexes and SYMPTOM_NAMES as a key-value pair and prediction_classes and ORIGINAL_PROGNOSIS as the second key-value pair.
### 5. Training.csv
- It contains the data I used to train the machine learning models.

### 6. Testing.csv
- it contains the dataset I used to test the machine learning models.

### 7. model.pkl files
These pkl files contain the five trained machine learning models which I created using the joblib library. They are:

    i. svm_model.pkl
    ii. naive_bayes_model.pkl
    iii. decision_tree_model.pkl
    iv. k_nearest_neighbors_model.pkl
    v. random_forest_model.pkl
