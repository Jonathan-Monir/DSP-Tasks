import streamlit as st
import math
import numpy as np
if "input_signals" in st.session_state:
    input_signals = st.session_state["input_signals"]
    
else:
    st.warning("Please add input signals")
    
if "output_signals" in st.session_state:
    output_signals = st.session_state["output_signals"]

else:
    st.warning("Please add output signals")

signal1 = input_signals[0][1]
signal2 = input_signals[1][1]

operation = st.selectbox("select operation",["normalize cross correlation","time delay","files"])
def calculate_average(values):
    if not values:
        return None
    return sum(values) / len(values)

def classify_signal(average, total_average_up,total_average_down):
    if abs(average-total_average_up)<abs(average-total_average_down):
        return "Up"
    else:
        return "Down"

def Compare_Signals(file_name,Your_indices,Your_samples):      
    expected_indices=[]
    expected_samples=[]
    with open(file_name, 'r') as f:
        line = f.readline()
        line = f.readline()
        line = f.readline()
        line = f.readline()
        while line:
            # process line
            L=line.strip()
            if len(L.split(' '))==2:
                L=line.split(' ')
                V1=int(L[0])
                V2=float(L[1])
                expected_indices.append(V1)
                expected_samples.append(V2)
                line = f.readline()
            else:
                break
    print("Current Output Test file is: ")
    print(file_name)
    print("\n")
    if (len(expected_samples)!=len(Your_samples)) and (len(expected_indices)!=len(Your_indices)):
        print("Shift_Fold_Signal Test case failed, your signal have different length from the expected one")
        return
    for i in range(len(Your_indices)):
        if(Your_indices[i]!=expected_indices[i]):
            print("Shift_Fold_Signal Test case failed, your signal have different indicies from the expected one") 
            return
    for i in range(len(expected_samples)):
        if abs(Your_samples[i] - expected_samples[i]) < 0.01:
            continue
        else:
            print("Correlation Test case failed, your signal have different values from the expected one") 
            return
    st.success("Correlation Test case passed successfully")


def normalized_cross_correlation(signal1, signal2):
    N = len(signal1)
    
    multiplied_signals = [signal1 * signal2 for signal1, signal2 in zip(signal1, signal2)]
    
    summed_result = sum(multiplied_signals) / N
    
    powered_signal1 = [signal1**2 for signal1 in signal1]
    powered_signal2 = [signal2**2 for signal2 in signal2]
    
    powered_result = sum(powered_signal1) * sum(powered_signal2)
    
    result = summed_result / ((1/N) * np.sqrt(powered_result))

    return result
normalized_signal = []
for _ in range(len(signal2)):
    result = normalized_cross_correlation(signal1, signal2)
    normalized_signal.append(result)
    first_element = signal2.pop(0)
    signal2.append(first_element)
    
if st.button("Submit"):
    if operation == "normalize cross correlation":
        
        indices = range(0,len(signal1))
        st.write("indices: ",list(indices))
        st.write("result: ",normalized_signal)
        Compare_Signals(r"files\task 7\Task Files\Point1 Correlation\CorrOutput.txt",indices,normalized_signal)
        
    elif operation == "time delay":
        
        # Find the time delay
        time_delay_index = np.argmax(normalized_signal)
        st.write(normalized_signal,time_delay_index)
        time_delay = (abs(time_delay_index - len(signal2)) + 1) / 100  # Assuming Fs=100

        st.write("Time Delay:", time_delay)
if operation == "files":
    # Get the number of known signals from the user
    num_known_signals = st.number_input("Enter the number of known signals (labeled as Up or Down)", min_value=0, value=0, step=1)

    known_signals = []
    for i in range(num_known_signals):
        uploaded_file = st.file_uploader(f"Choose known signal file {i+1}", type="txt")
        classification = st.selectbox(f"Select classification for known Signal {i+1}", ["Up", "Down"])
        known_signals.append({"file": uploaded_file, "classification": classification})

    # Calculate average for known signals
    up_values = []
    down_values = []

    for i, signal in enumerate(known_signals):
        st.write(f"### Known Signal {i+1} ({signal['classification']}):")

        if signal['file'] is not None:
            content = signal['file'].read().decode("utf-8")
            values = [float(value.strip()) for value in content.splitlines() if value.strip().replace('.', '', 1).isdigit()]

            average = calculate_average(values)
            # st.write("#### File Content:")
            # st.code(content)

            if average is not None:
                # Add the average value to the appropriate array
                if signal['classification'] == "Up":
                    up_values.append(average)
                elif signal['classification'] == "Down":
                    down_values.append(average)
            else:
                st.warning(f"No valid numerical values found in Known Signal {i+1}.")

    # Display total averages for up and down values
    total_average_up = calculate_average(up_values)
    total_average_down = calculate_average(down_values)

    st.write("### Summary for Known Signals:")
    st.write("Up Values:", up_values)
    st.write("Down Values:", down_values)
    st.write("Total Average for Up Values:", total_average_up)
    st.write("Total Average for Down Values:", total_average_down)

    # Get the number of signals to classify
    num_unknown_signals = st.number_input("Enter the number of signals to classify", min_value=1, value=1, step=1)

    # Allow the user to upload and classify additional text files
    for i in range(num_unknown_signals):
        new_file = st.file_uploader(f"Upload text file {i+1} for classification", type="txt")
        if new_file is not None:
            new_content = new_file.read().decode("utf-8")
            new_values = [float(value.strip()) for value in new_content.splitlines() if value.strip().replace('.', '', 1).isdigit()]

            new_average = calculate_average(new_values)
            # st.write(f"### New File Content {i+1}:")
            # st.code(new_content)

            if new_average is not None:
                # Classify and add the new average value to the appropriate array
                classification = classify_signal(new_average, total_average_up,total_average_down)
                st.success(f"Classified as '{classification}'. Average: {new_average:.2f}")
                if classification == "Up":
                    up_values.append(new_average)
                else:
                    down_values.append(new_average)
            else:
                st.warning(f"No valid numerical values found in the new file {i+1}.")

    # Display updated total averages for up and down values
    st.write("### Summary after Classifying New Signals:")
    # st.write("Up Values:", up_values)
    # st.write("Down Values:", down_values)
    st.write("Updated Total Average for Up Values:", calculate_average(up_values))
    st.write("Updated Total Average for Down Values:", calculate_average(down_values))
