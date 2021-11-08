from predict_tools import take_input, create_new_symptoms, create_input_data, predict_disease, np

import tkinter as tk

# This function creates the Graphical User Interface.
def create_gui():
    window = tk.Tk()

    window.title("Disease Predictor")

    window.geometry("400x200")

    # Creating label for instructions.
    label1 = tk.Label(window, text=
    "Hello there friend, Heard you were feeling a bit under the weather. \n Tell me what the problem is and please feel free to list as many symptoms as you can. \n The more symptoms you enter, the more accurate your diagnosis will be."
    , fg="black")
    label1.grid(row=0, column=0, sticky = "n")

    # Getting the input text from the user
    input_text = tk.StringVar("")
    input_text.set("")

    output_text = tk.StringVar("")
    output_text.set("Result: ")
    
    # Creating button that gets the predictions.
    b1 = tk.Button(window, text="Get Predictions", command=lambda:on_predict(input_text, output_text))
    b1.grid(row=10, column=0)

    # Creates entry box for the user to enter symptoms.
    e1 = tk.Entry(window, textvariable = input_text, bg = "white")
    e1.grid(row=3, column=0, sticky="w")
    
    # Creating button that clears the entry box.
    b2 = tk.Button(window, text="Clear", command = lambda:clear_textbox(e1))
    b2.grid(row=4, column=0, sticky="w")

    # Creating the label that displays the diagnosis.
    result_label = tk.Label(window, textvariable=output_text, fg = "black", bg = "white")
    result_label.grid(row=12, column=0, sticky="w")

    return window

# This function predicts the disease and returns it as output.
def on_predict(input_text, output_text):
    input_symptoms = take_input(input_text.get())

    # Calling the create_new_symptoms function to create a list containing the symptoms which the user entered. 
    new_symptoms_list = create_new_symptoms(input_symptoms)

    # Calling the create_input_data function which creates a numpy array with the users inputs assigned to 1 and the remaining assigned to 0. 
    input_data = create_input_data(new_symptoms_list)

    # Checking if the user entered valid inputs.
    if np.max(input_data) == np.min(input_data):
        output = "Your entries are not valid please try again."
    else:
        # Calling the predict disease function which uses the ml models to predict the disease based on the data in input_data.
        diagnosis = predict_disease(input_data)
        output = f"Based on your symptoms you might have {diagnosis}. \n Please seek medical assistance immediately."
    
    output_text.set(output)

# function that clears the entry box.
def clear_textbox(e1):
    e1.delete(0, 'end')


def main():
    # Calling the create_gui function
    window = create_gui()

    window.mainloop()

if __name__ == "__main__":
    main()