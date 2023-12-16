import streamlit as st
import numpy as np
import math
import matplotlib.pyplot as plt


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
