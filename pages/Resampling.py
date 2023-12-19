import streamlit as st
import math
import matplotlib.pyplot as plt
import numpy as np
from fractions import Fraction
import plotly.express as px
import pandas as pd

class ApplyFilter:
    def __init__(self,stop_band,transition_width,pass_band_frequency,Fs):
        self.stop_band = stop_band
        self.transition_width = transition_width / Fs
        self.pass_band_frequency = (pass_band_frequency + (transition_width/2)) / Fs
        self.Fs = Fs

        self.window_name = self.window_define()
        self.N = int(np.ceil(self.number_of_samples_define()))
        self.X = range(int((self.N-1)/2 * -1), int((self.N-1)/2) +1)
        self.W = self.apply_function_W()
        self.H = self.apply_function_H()

        self.Y = self.apply_function_Y()
        
    def window_define(self):
        if self.stop_band <= 21:
            return "Rectangular"
        elif self.stop_band <= 44:
            return "Hanning"
        elif self.stop_band <= 53:
            return "Hamming"
        elif self.stop_band <= 74:
            return "Blackman"
        else:
            return 0


    def number_of_samples_define(self):
        if self.window_name == "Rectangular":
            transition_width_factor = 0.9
        elif self.window_name == "Hanning":
            transition_width_factor = 3.1
        elif self.window_name == "Hamming":
            transition_width_factor = 3.3
        elif self.window_name == "Blackman":
            transition_width_factor = 5.5

        return transition_width_factor/self.transition_width

    def apply_function_W(self):
        if self.window_name == "Rectangular":
            return {n:1 for n in self.X}
        elif self.window_name == "Hanning":
            return {n:(0.5 + 0.5 * math.cos(2*math.pi*n /self.N)) for n in self.X}
        elif self.window_name == "Hamming":
            return {n:(0.54 + 0.46 * math.cos(2*math.pi*n / self.N)) for n in self.X}
        elif self.window_name == "Blackman":
            return {n:(0.42 + 0.5 * math.cos(2*math.pi*n / (self.N-1)) + 0.08 * math.cos(4*math.pi*n / (self.N-1))) for n in self.X}
    
    def apply_function_H(self):
        H = {n:2 * self.pass_band_frequency * math.sin(n*2*math.pi*self.pass_band_frequency)/(2*math.pi*self.pass_band_frequency*n) for n in self.X if n !=0}
        H[0] = 2*self.pass_band_frequency
        
        return H

    def apply_function_Y(self):
        
        Y = [self.H[n] * self.W[n] for n in self.X]
        return Y

def create_plotly_plot(x, y, plot_name):
    df = pd.DataFrame({'x': x, 'y': y})
    fig = px.line(df, x='x', y='y', title=plot_name)
    return fig

def convolve(input_signal, filter_kernel, start_value):
    input_len = len(input_signal)
    filter_len = len(filter_kernel)

    if input_len == 0 or filter_len == 0:
        raise ValueError("Input signals cannot be empty for convolution.")

    output_len = input_len + filter_len - 1

    output_signal = [0] * output_len

    for i in range(output_len):
        for j in range(filter_len):
            if i - j >= 0 and i - j < input_len:
                output_signal[i] += input_signal[i - j] * filter_kernel[j]

    output_indices = list(range(start_value, start_value + output_len))

    return output_indices, output_signal

def Downsample(signals,Y, L, start_value,M):
    if L < 2:
        signals = convolve(signals,Y,start_value)[1]
    
    downsampled_data = []
    for i in range(0, len(signals), M):
        downsampled_data.extend(signals[i:i+1])
    
    return downsampled_data

def Upsample(signals,Y, L, start_value):
    upsampled_signals = []
    for index, signal in enumerate(signals):
        upsampled_signals.append(signal)
        
        if index == len(signals) - 1:
            break

        for _ in range(L-1):
            upsampled_signals.append(0)
    upsampled_signals = convolve(upsampled_signals,Y,start_value)
    return upsampled_signals

def has_decimal(float_number):
    return float_number != int(float_number)

def parse_signal_data(data):
    x_values, y_values = [], []
    for line in data:
        parts = line.strip().split()
        if len(parts) == 2:
            x, y = parts
            try:
                # Convert to float and append to the lists
                x_values.append(float(x))
                y_values.append(float(y))
            except ValueError:
                # Handle invalid data here (e.g., skip or log it)
                pass
        elif len(parts) > 2:
            # If there are more than two parts, try to extract the first two as x and y
            x, y = parts[:2]
            try:
                x_values.append(float(x))
                y_values.append(float(y))
            except ValueError:
                # Handle invalid data here (e.g., skip or log it)
                pass
    return x_values, y_values

def Compare_Signals(file_name, Your_indices, Your_samples):    
    expected_indices = []
    expected_samples = []
    with open(file_name, 'r') as f:
        line = f.readline()
        line = f.readline()
        line = f.readline()
        line = f.readline()
        while line:
            # process line
            L=line.strip()
            if len(L.split(' ')) == 2:
                L = line.split(' ')
                V1 = int(L[0])
                V2 = float(L[1])
                expected_indices.append(V1)
                expected_samples.append(V2)
                line = f.readline()
            else:
                break
    print("Current Output Test file is: ")
    print(file_name)
    print("\n")
    if (len(expected_samples) != len(Your_samples)) or (len(expected_indices) != len(Your_indices)):
        st.write(len(expected_indices), len(Your_indices))
        st.write("Test case failed, your signal has a different length from the expected one")
        return
    for i in range(len(Your_indices)):
        
        if Your_indices[i] != expected_indices[i]:
            st.write("Test case failed, your signal has different indices from the expected one") 
            return
    for i in range(len(expected_samples)):
        if abs(Your_samples[i] - expected_samples[i]) < 0.01:
            continue
        else:
            st.write("Test case failed, your signal has different values from the expected one at: ",i) 
            return
    st.write("Test case passed successfully")

st.subheader("Input Signals")

num_signals = st.number_input("How many signals do you want to analyze?", min_value=1, value=1, key="num_signals")

input_signals = []
output_signals = []

for i in range(num_signals):
    signal_file = st.file_uploader(f'Upload Signal {i + 1}', type=['txt'], key=f'signal_uploader_{i}')
    output_signal_file = st.file_uploader(f'Upload Output Signal {i + 1}', type=['txt'], key=f'output_signal_uploader_{i}')

    if signal_file is not None:
        uploaded_data = signal_file.read().decode('utf-8')
        lines = uploaded_data.split('\n')
        x_values, y_values = parse_signal_data(lines)
        
        input_signals.append((x_values, y_values))
    
    if output_signal_file is not None:
        uploaded_data = output_signal_file.read().decode('utf-8')
        lines = uploaded_data.split('\n')
        output_x_values, output_y_values= parse_signal_data(lines)

        # Ensure that the values are correctly converted to float when parsing the data
        output_signals.append((output_x_values, output_y_values))

Fs = st.number_input("Sampling frequencty",0.0,100000.0,step=10.0,value=8.0)
stop_band = st.number_input("stop band",0.0,100000.0,step=0.0,value=50.0)
pass_band_frequency = st.number_input("pass band frequency",0.0,100000.0,step=0.5,value=1.5)
Transition_width = st.number_input("Transition width",0.0,50000.0,step=0.25,value=0.5)

# Fs = 8000
# stop_band = 50
# pass_band_frequency = 1500
# Transition_width = 500

applied = ApplyFilter(stop_band, Transition_width,pass_band_frequency,Fs)

if st.checkbox("upsampling or downsampling"):
    L = st.number_input("Upsampling (L)",0.0,100000.0,step=1.0,value=1.0)
    M = st.number_input("Downsampling (M)",0.0,100000.0,step=1.0,value=1.0)
    Resampled_signals = input_signals[0][1]
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
        Resampled_signals = Upsample(input_signals[0][1], applied.Y, L, applied.X[0])[1]

    if M > 1:
        Resampled_signals = Downsample(Resampled_signals, applied.Y, L, applied.X[0],M)
        
    indices = range(applied.X[0], applied.X[0] + len(Resampled_signals))
    if M == 0 and L == 0:
        st.error("M and L shouldn't be = 0")
test_path = st.selectbox("please select the task file to compare signals: ", ["Testcase 1\Sampling_Down.txt","Testcase 2\Sampling_Up.txt","Testcase 3\Sampling_Up_Down.txt"])
filepath = r"files\Practical task\Practical task 1\Sampling test cases" +"\\" + test_path
st.write(Resampled_signals)
Compare_Signals(filepath,indices,Resampled_signals)

# if test_path == "Testcase 3\Sampling_Up_Down.txt":
#     st.error("error due to extra space in files")

if st.checkbox("Show graph"):

    fig = create_plotly_plot(indices, Resampled_signals, test_path.split(".")[0].split("\\")[1])
    st.plotly_chart(fig)