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

operation = st.selectbox("select operation",["normalize cross correlation","time delay"])

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
if st.button("Submit"):
    if operation == "normalize cross correlation":
        normalized_signal = []
        for _ in range(len(signal2)):
            result = normalized_cross_correlation(signal1, signal2)
            normalized_signal.append(result)
            first_element = signal2.pop(0)
            signal2.append(first_element)
        indices = range(0,len(signal1))
        st.write("indices: ",list(indices))
        st.write("result: ",normalized_signal)
        Compare_Signals(r"files\task 7\Task Files\Point1 Correlation\CorrOutput.txt",indices,normalized_signal)
            