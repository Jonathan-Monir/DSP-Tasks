import streamlit as st
import os
from fractions import Fraction
import math
import numpy as np
st.subheader("files")

file_a = st.text_input('Enter file path A',value = r"files\Practical task 2\A")
file_b = st.text_input('Enter file path B',value = r"files\Practical task 2\B")
# file_tests = st.text_input('Enter file path tests',value = r"files\Practical task 2\Test Folder")

st.subheader("Resampling")

L = st.number_input("Upsampling (L)",0.0,100000.0,step=1.0,value=1.0)
M = st.number_input("Downsampling (M)",0.0,100000.0,step=1.0,value=1.0)

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
    return [0] * len(data)  # Handle edge case where all values are equal
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
    # upsampled_signals = convolve(upsampled_signals,Y,start_value)
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
    A_signals = path_to_list(filepath_A)
    B_signals = path_to_list(filepath_B)

    # task 1) filter the signal


    # task 2) resampling
    # A_signals = [resample(signal,L,M) for signal in A_signals]
    # B_signals = [resample(signal,L,M) for signal in B_signals]

    # task 3) remove DC

    # task 4) Normalize
    # A_signals = [normalize(signal) for signal in A_signals]
    # B_signals = [normalize(signal) for signal in B_signals]

    # task 5) auto correlation

    # task 6) get only the needed coefficients

    # task 7) DCT
    # A_signals = [dct(signal) for signal in A_signals]
    # B_signals = [dct(signal) for signal in B_signals]

ECG(file_a,file_b,0,0,0,0)