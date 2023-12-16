import streamlit as st
import numpy as np
import math
import matplotlib.pyplot as plt
import numpy as np

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
        if self.stop_band < 21:
            return "Rectangular"
        elif self.stop_band < 44:
            return "Hanning"
        elif self.stop_band < 53:
            return "Hamming"
        elif self.stop_band < 74:
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
            transition_width_factor = 3.5

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

def parse_signal_data(data):
    x_values, y_values, z_values = [], [], []
    for line in data:
        if ',' in line:
            parts = line.strip().split(',')
        else:
            parts = line.strip().split()
        if len(parts) == 2:
            x, y = parts
            x = x.replace('f', '')
            y = y.replace('f', '')
            try:
                x_values.append(float(x))
                y_values.append(float(y))
            except ValueError:
                # Handle invalid data here (e.g., skip or log it)
                pass
        elif len(parts) == 3:
            x, y, z = parts
            x = x.replace('f', '')
            y = y.replace('f', '')
            z = z.replace('f', '')
            try:
                x_values.append(float(x))
                y_values.append(float(y))
                z_values.append(float(z))
            except ValueError:
                # Handle invalid data here (e.g., skip or log it)
                pass
    return x_values, y_values, z_values

    
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
        x_values, y_values, z_values = parse_signal_data(lines)
        
        input_signals.append((x_values, y_values, z_values))
    
    if output_signal_file is not None:
        uploaded_data = output_signal_file.read().decode('utf-8')
        lines = uploaded_data.split('\n')
        output_x_values, output_y_values, output_z_values = parse_signal_data(lines)

        # Ensure that the values are correctly converted to float when parsing the data
        output_signals.append((output_x_values, output_y_values, output_z_values))

pass_band_frequency = st.number_input("pass band frequency",0.0,100000.0,step=0.5,value=1.5)
Transition_width = st.number_input("Transition width",0.0,50000.0,step=0.25,value=0.5)
stop_band = st.number_input("stop band",0.0,100000.0,step=0.0,value=50.0)
Fs = st.number_input("Sampling frequencty",0.0,100000.0,step=10.0,value=8.0)

applied = ApplyFilter(stop_band, Transition_width,pass_band_frequency,Fs)
st.write(applied.X)
st.write(applied.Y)
