import streamlit as st
import os
file_a = st.text_input('Enter file path A')
file_b = st.text_input('Enter file path B')
file_tests = st.text_input('Enter file path tests')

def path_to_list(path):
    # Initialize an empty list to store all the lists from text files
    all_lists = []

    # Use the os module to list all files in the given path
    import os

    # Loop through each file in the directory
    for filename in os.listdir(path):
        # Check if the file has a .txt extension
        if filename.endswith(".txt"):
            file_path = os.path.join(path, filename)
            
            # Read the content of the file and convert it into a list of numbers
            with open(file_path, 'r') as file:
                numbers_list = [float(line.strip()) for line in file]

            # Append the list of numbers to the big list
            all_lists.append(numbers_list)

    return all_lists

def ECG(filepath_A, filepath_B, file_tests, Fs, min_F, max_F, new_Fs):
    A_signals = path_to_list(filepath_A)
    B_signals = path_to_list(filepath_B)
    test_signals = path_to_list(file_tests)

ECG(file_a,file_b,file_tests,0,0,0,0)