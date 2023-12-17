import streamlit as st
import os
from fractions import Fraction
import math
import numpy as np
from ttt import design_fir_filter, convolve
import plotly.express as px
import pandas as pd

st.subheader("files")

file_a = st.text_input('Enter file path A',value = r"files\Practical task 2\A")
file_b = st.text_input('Enter file path B',value = r"files\Practical task 2\B")
# file_tests = st.text_input('Enter file path tests',value = r"files\Practical task 2\Test Folder")

Fs = st.number_input("Fs", 1,200,value = 10)
new_Fs = st.number_input("new Fs", 1,200,value = 30)
st.subheader("Resampling")

L = st.number_input("Upsampling (L)",0.0,100000.0,step=1.0,value=1.0)
M = st.number_input("Downsampling (M)",0.0,100000.0,step=1.0,value=1.0)


def create_plotly_plot(x, y, plot_name):
    df = pd.DataFrame({'x': x, 'y': y})
    fig = px.line(df, x='x', y='y', title=plot_name)
    return fig


def normalized_cross_correlation(signal1, signal2):
    N = len(signal1)
    
    multiplied_signals = [signal1 * signal2 for signal1, signal2 in zip(signal1, signal2)]
    
    summed_result = sum(multiplied_signals) / N
    
    powered_signal1 = [signal1**2 for signal1 in signal1]
    powered_signal2 = [signal2**2 for signal2 in signal2]
    
    powered_result = sum(powered_signal1) * sum(powered_signal2)
    
    result = summed_result / ((1/N) * np.sqrt(powered_result))


    return result

def RemoveDc(input_signals):
    # Extract y_values from each tuple
    y_values_list = [signal[1] for signal in input_signals]

    # Calculate the sum of y_values across all signals
    total_sum = sum([sum(y_values) for y_values in y_values_list])

    # Calculate the average
    average = total_sum / sum(len(y_values) for y_values in y_values_list)

    # Subtract the average from each y_values list
    result = [[x, [y - average for y in y_values]] for x, y_values in input_signals]

    return result
    
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

def normalize(data):
    
  min_val = min(data)
  max_val = max(data)
  if min_val == max_val:
    return [0] * len(data) 
  return [(x - min_val) / (max_val - min_val) * 2 - 1 for x in data]

def has_decimal(float_number):
    return float_number != int(float_number)

# def dct(input_list):
#     N = len(input_list)
#     result = []
#     for k in range(N):
#         sum_result = 0
#         for n in range(N):
#             sum_result += input_list[n] * math.cos(math.pi / (4 * N) * (2 * n - 1) * (2 * k - 1))
            
#         result.append(math.sqrt(2 / N) * sum_result)
        
#     return result


def dct(input_list):

  N = len(input_list)
  data = np.array(list(input_list))
  # Use pre-computed cosine matrix for efficiency
  cosine_matrix = np.cos(np.pi * np.arange(N)[:, None] * (2 * np.arange(N) + 1) / (4 * N))
  # Perform the DCT using matrix multiplication
  dct_coefficients = np.sqrt(2 / N) * cosine_matrix @ data
  return dct_coefficients.tolist()


def Downsample(signals, L,M):
    # if L < 2:
        # signals = convolve(signals,Y,start_value)[1]
    
    downsampled_data = []
    for i in range(0, len(signals), M):
        downsampled_data.extend(signals[i:i+1])
    
    return downsampled_data

def Upsample(signals, L):
    upsampled_signals = []
    for index, signal in enumerate(signals):
        upsampled_signals.append(signal)
        
        if index == len(signals) - 1:
            break

        for _ in range(L-1):
            upsampled_signals.append(0)
    upsampled_signals = convolve(upsampled_signals,Y,start_value)
    return upsampled_signals

def resample(signals,L,M):

    Resampled_signals = signals
    if has_decimal(L):
        fraction_object = Fraction(L)
        L = fraction_object.numerator
        M = M * fraction_object.denominator
    elif has_decimal(M):
        fraction_object = Fraction(M)
        M = fraction_object.numerator
        L = L * fraction_object.denominator

    L = int(L)
    M = int(M)
    
    if L > 1:
        Resampled_signals = Upsample(signals, L)[1]

    if M > 1:
        Resampled_signals = Downsample(Resampled_signals, L,M)
    return Resampled_signals

def ECG(filepath_A, filepath_B, Fs, min_F, max_F, new_Fs):

    InputStopBandAttenuation = 50
    InputTransitionBand = 500

    A_signals = path_to_list(filepath_A)
    B_signals = path_to_list(filepath_B)
    
    # task 1) filter the signal
    fir_coefficients, start_value = design_fir_filter("bandpass", Fs, InputStopBandAttenuation, min_F, max_F, InputTransitionBand)

    A_signals = [convolve(signal, [val for _, val in fir_coefficients], start_value)[1] for signal in A_signals]
    B_signals = [convolve(signal, [val for _, val in fir_coefficients], start_value)[1] for signal in B_signals]

    A_signals_indices = [convolve(signal, [val for _, val in fir_coefficients], start_value)[0] for signal in A_signals]
    B_signals_indices = [convolve(signal, [val for _, val in fir_coefficients], start_value)[0] for signal in B_signals]
    
    
    # task 2) resampling
    A_signals = [resample(signal,L,M) for signal in A_signals if new_Fs > 2*np.max(signal)]
    B_signals = [resample(signal,L,M) for signal in B_signals if new_Fs > 2*np.max(signal)]

    st.plotly_chart(create_plotly_plot(A_signals_indices[1], A_signals[1],'da'))

    # task 3) remove DC
    A_signals = [RemoveDc(signal) for signal in A_signals]
    B_signals = [RemoveDc(signal) for signal in B_signals]

    # task 4) Normalize
    A_signals = [normalize(signal) for signal in A_signals]
    B_signals = [normalize(signal) for signal in B_signals]

    # task 5) auto correlation
    # normalized_cross_correlation

    # task 6) get only the needed coefficients

    # task 7) DCT
    # A_signals = [dct(signal) for signal in A_signals]
    # B_signals = [dct(signal) for signal in B_signals]

ECG(file_a,file_b,Fs,100,250,new_Fs)