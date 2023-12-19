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
file_tests = st.text_input('Enter file path tests',value = r"files\Practical task 2\Test Folder")

Fs = st.number_input("Fs", 1,200,value = 10)
new_Fs = st.number_input("new Fs", 1,2000,value = 30)
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

def correlation_calculator(signal1,signal2):
    normalized_signal = []
    for i in range(len(signal2)):
        print(i)
        result = normalized_cross_correlation(signal1, signal2)
        normalized_signal.append(result)
        signal2 = np.roll(signal2, -1)  
    # Convert the list to a NumPy array if needed
    normalized_signal = np.array(normalized_signal)
    return normalized_signal
# def dft(input_signal):
#     N = len(input_signal)
#     magnitude = np.zeros(N)
#     phase = np.zeros(N)

#     for k in range(N):
#         real_part = 0.0
#         imag_part = 0.0

#         for n in range(N):
#             angle = 2 * np.pi * k * n / N
#             real_part += input_signal[n] * np.cos(angle)
#             imag_part -= input_signal[n] * np.sin(angle)

#         magnitude[k] = np.sqrt(real_part**2 + imag_part**2)
#         phase[k] = np.arctan2(imag_part, real_part)

#     return magnitude, phase

# def convolve_freq_domain(signal1, signal2):
#         # Calculate the size for zero-padding
#         size = len(signal1) + len(signal2) - 1

#         # Zero-pad the input signals
#         signal1_padded = np.pad(signal1, (0, size - len(signal1)))
#         signal2_padded = np.pad(signal2, (0, size - len(signal2)))

#         # Compute DFT of input signals
#         mag1, phase1 = dft(signal1_padded)
#         mag2, phase2 = dft(signal2_padded)

#         # Perform element-wise multiplication in the frequency domain
#         mag_result = mag1 * mag2
#         phase_result = phase1 + phase2

#         # Perform IDFT on the result to get the convolution in time domain
#         convolution_result = idft(mag_result, phase_result)

#         return convolution_result

from scipy.fft import fft
def dft(input_signal):
    fft_result = fft(input_signal)
    magnitude = np.abs(fft_result)
    phase = np.angle(fft_result)
    return magnitude, phase

from scipy.fft import ifft
def idft(amplitude, phase):
    # Combine amplitude and phase to create a complex signal
    complex_signal = amplitude * np.exp(1j * np.array(phase))
    
    # Perform the IFFT
    result = ifft(complex_signal)
    return result

def correlation(signal1,signal2):
    
    if len(signal1) != len(signal2):
        
        
        # Calculate the size for zero-padding
        size = len(signal1) + len(signal2) - 1

        # Zero-pad the input signals
        signal1 = np.pad(signal1, (0, size - len(signal1)))
        signal2 = np.pad(signal2, (0, size - len(signal2)))
    
    X = dft(signal1)
    Y = dft(signal2) 
    
    Y = list(Y)
    Y[1] = np.array(Y[1]) * -1
    Y = tuple(Y)
    
    mag1, phase1 = X
    mag2, phase2 = Y
    
    mag_result = mag1 * mag2
    phase_result = phase1 + phase2

    # Perform IDFT on the result to get the convolution in time domain
    convolution_result = idft(mag_result, phase_result)
    
    # extract only the magnitude of convolution result where it's type is numpy.complex128
    convolution_result = np.abs(convolution_result)

    return convolution_result

def RemoveDc(input_signals):
    # Extract y_values from each tuple
    # y_values_list = [signal[1] for signal in input_signals]
    # st.write(input_signals)
    # Calculate the sum of y_values across all signals
    total_sum = sum([sum(y_values) for y_values in input_signals])
    
    # Calculate the average
    average = total_sum / sum(len(y_values) for y_values in input_signals)
    
    # Subtract the average from each y_values list
    result = [[y - average for y in signal] for signal in input_signals]
    # result = [[x, [y - average for y in y_values]] for x, y_values in indices, input_signals]

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



def convolve_freq_domain(signal1, signal2):
    # Calculate the size for zero-padding
    size = len(signal1) + len(signal2) - 1

    # Zero-pad the input signals
    signal1_padded = np.pad(signal1, (0, size - len(signal1)))
    signal2_padded = np.pad(signal2, (0, size - len(signal2)))

    # Compute DFT of input signals
    mag1, phase1 = dft(signal1_padded)
    mag2, phase2 = dft(signal2_padded)

    # Perform element-wise multiplication in the frequency domain
    mag_result = mag1 * mag2
    phase_result = phase1 + phase2

    # Perform IDFT on the result to get the convolution in time domain
    convolution_result = idft(mag_result, phase_result)

    return convolution_result

    
def dct(input_list):

  N = len(input_list)
  data = np.array(list(input_list))
  # Use pre-computed cosine matrix for efficiency
  cosine_matrix = np.cos(np.pi * np.arange(N)[:, None] * (2 * np.arange(N) + 1) / (4 * N))
  # Perform the DCT using matrix multiplication
  dct_coefficients = np.sqrt(2 / N) * cosine_matrix @ data
  return dct_coefficients.tolist()


def Downsample(signals, L, M, Y, start_value):
    if L < 2:
        signals = convolve(signals,Y,start_value)[1]
    
    downsampled_data = []
    for i in range(0, len(signals), M):
        downsampled_data.extend(signals[i:i+1])
    
    return downsampled_data

def Upsample(signals, L, Y, start_value):
    
    upsampled_signals = []
    for index, signal in enumerate(signals):
        upsampled_signals.append(signal)
        
        if index == len(signals) - 1:
            break

        for _ in range(L-1):
            upsampled_signals.append(0)
            
    upsampled_signals = convolve(upsampled_signals,Y,start_value)[1]
    
    
    return upsampled_signals

def resample(signals,L,M, Y, start_value):

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
        Resampled_signals = Upsample(signals, L, Y, start_value)
    
    if M > 1:
        Resampled_signals = Downsample(Resampled_signals, L,M, Y, start_value)
    
    return Resampled_signals

def ECG(filepath_A, filepath_B, filepath_test, Fs, min_F, max_F, new_Fs):

    InputStopBandAttenuation = 50
    InputTransitionBand = 500

    A_signals = path_to_list(filepath_A)
    B_signals = path_to_list(filepath_B)
    test_signals = path_to_list(filepath_test)
    # task 1) filter the signal
    fir_coefficients, start_value = design_fir_filter("bandpass", Fs, InputStopBandAttenuation, min_F, max_F, InputTransitionBand)

    A_signals = [convolve(signal, [val for _, val in fir_coefficients], start_value)[1] for signal in A_signals]
    B_signals = [convolve(signal, [val for _, val in fir_coefficients], start_value)[1] for signal in B_signals]

    A_signals_indices = [convolve(signal, [val for _, val in fir_coefficients], start_value)[0] for signal in A_signals]
    B_signals_indices = [convolve(signal, [val for _, val in fir_coefficients], start_value)[0] for signal in B_signals]
    
    st.plotly_chart(create_plotly_plot(A_signals_indices[0], A_signals[0],'first file in A after applying filter'))

    
    # task 2) resampling
    fir_coefficients, start_value = design_fir_filter("lowpass", 8000, 50, 1500, max_F, 500)
    
    for index, signal in enumerate(A_signals):
        if new_Fs >= 2*np.max(signal):
            
            A_signals[index] = resample(signal,L,M,[val for _, val in fir_coefficients],start_value)
            A_signals_indices[index] = list(range(start_value, start_value+len(A_signals[index])))
            
    for index, signal in enumerate(B_signals):
        if new_Fs >= 2*np.max(signal):
            B_signals[index] = resample(signal,L,M,[val for _, val in fir_coefficients],start_value)
            B_signals_indices[index] = list(range(start_value, start_value+len(B_signals[index])))
            
    st.plotly_chart(create_plotly_plot(A_signals_indices[0], A_signals[0],'first file in A after applying resampling'))
    

    # compressed_A_signals = [(A_signals_indices[index],A_signals)for index in range(len(A_signals_indices))]
    # compressed_B_signals = [(B_signals_indices[index],B_signals)for index in range(len(B_signals_indices))]
    # st.write(A_signals_indices,"\n")
    # print([signal[1] for signal in compressed_A_signals])
    # task 3) remove DC

    A_signals = RemoveDc(A_signals)
    B_signals = RemoveDc(B_signals)

    st.plotly_chart(create_plotly_plot(A_signals_indices[0], A_signals[0],'first file in A after applying RemoveDc'))

    # task 4) Normalize
    A_signals = [normalize(signal) for signal in A_signals]
    B_signals = [normalize(signal) for signal in B_signals]

    st.plotly_chart(create_plotly_plot(A_signals_indices[0], A_signals[0],'first file in A after applying normalize'))

    # task 6) get only the needed coefficients
    A_signals = [signal[1250:1500] for signal in A_signals]
    B_signals = [signal[1250:1500] for signal in B_signals]

    A_signals_indices = [signal[1250:1500] for signal in A_signals_indices]
    B_signals_indices = [signal[1250:1500] for signal in B_signals_indices]
    st.plotly_chart(create_plotly_plot(A_signals_indices[0], A_signals[0],'first file in A after needed coefficients'))
    
    # task 5) auto correlation
    # normalized_cross_correlation
    A_signals = [correlation_calculator(signal,signal) for signal in A_signals]
    B_signals = [correlation_calculator(signal,signal) for signal in B_signals]
    st.plotly_chart(create_plotly_plot(A_signals_indices[0], A_signals[0],'first file in A after applying correlation'))

    # task 7) DCT
    A_signals = [dct(signal) for signal in A_signals]
    B_signals = [dct(signal) for signal in B_signals]
    st.plotly_chart(create_plotly_plot(A_signals_indices[0], A_signals[0],'first file in A after dct'))
    
    # task 8)

    A_signals = np.array(A_signals)
    B_signals = np.array(B_signals)

    A_signals = np.average(A_signals, axis=0)
    B_signals = np.average(B_signals, axis=0)

    st.write(correlation_calculator(A_signals,test_signals[0])[0])
    st.write(correlation_calculator(A_signals,test_signals[1])[0])
    st.write(correlation_calculator(B_signals,test_signals[1])[0])
    st.write(correlation_calculator(B_signals,test_signals[0])[0])

ECG(file_a,file_b,file_tests,Fs,100,250,new_Fs)