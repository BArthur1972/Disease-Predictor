from predict_tools import take_input, create_new_symptoms, create_input_data, predict_disease, np
import tkinter as tk


def on_predict(input_text, output_text):
    '''This function takes input text and output text, predicts the disease and sets the diagnosis to output text.

    Args:
        input_text: the symptoms which the user inputs.
        output_text: a variable which will be set to the predicted diagnosis.

    Returns:
        None
    '''
    input_symptoms = take_input(input_text.get())

    new_symptoms_list = create_new_symptoms(input_symptoms)

    input_data = create_input_data(new_symptoms_list)

    # Checking if the user entered valid inputs by comparing.
    if np.max(input_data) == np.min(input_data):
        output = "Your entries are not valid please try again."
    else:
        diagnosis = predict_disease(input_data)
        output = f"Based on your symptoms you might have {diagnosis}. \n Please seek medical assistance immediately."

    output_text.set(output)


def create_gui():
    '''This function creates the Graphical User Interface.
    Args: 
        Nothing

    Returns: 
        Returns the window containing all the widgets made in the function using tkinter.
    '''
    window = tk.Tk()
    window.title("Disease Predictor")
    window.geometry("400x200")

    intro_label = tk.Label(
        window,
        text=
        "Hello there friend, Heard you were feeling a bit under the weather. \n Tell me what the problem is and please feel free to list as many symptoms as you can. \n The more symptoms you enter, the more accurate your diagnosis will be.",
        fg="black")
    intro_label.grid(row=0, column=0, sticky="n")

    input_text = tk.StringVar("")
    input_text.set("")

    output_text = tk.StringVar("")
    output_text.set("Result: ")

    get_predictions_button = tk.Button(
        window,
        text="Get Predictions",
        command=lambda: on_predict(input_text, output_text))
    get_predictions_button.grid(row=10, column=0)

    entry_box = tk.Entry(window, textvariable=input_text, bg="white")
    entry_box.grid(row=3, column=0, sticky="w")

    clear_button = tk.Button(window,
                             text="Clear",
                             command=lambda: entry_box.delete(0, 'end'))
    clear_button.grid(row=4, column=0, sticky="w")

    result_label = tk.Label(window,
                            textvariable=output_text,
                            fg="black",
                            bg="white")
    result_label.grid(row=12, column=0, sticky="w")

    return window


def main():
    window = create_gui()

    window.mainloop()


if __name__ == "__main__":
    main()
