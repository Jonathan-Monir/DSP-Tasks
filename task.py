import streamlit as st
import numpy as np
from scipy.fft import fft
import math

def dft(input_signal):
    fft_result = fft(input_signal)
    magnitude = np.abs(fft_result)
    phase = np.angle(fft_result)
    return magnitude, phase

def format_samples(signal):
    return "\n".join([f"{x} {y}" for x, y in enumerate(signal)])

def parse_signal_data(data):
    x_values, y_values = [], []
    for line in data:
        parts = line.strip().split()
        if len(parts) == 2:
            x, y = parts
            try:
                x_values.append(float(x))
                y_values.append(float(y))
            except ValueError:
                # Handle invalid data here (e.g., skip or log it)
                pass
    return x_values, y_values

def SignalCompareAmplitude(SignalInput, SignalOutput):
    if len(SignalInput) != len(SignalOutput):
        return False
    else:
        for i in range(len(SignalInput)):
            if abs(SignalInput[i] - SignalOutput[i]) > 0.001:
                return False
    return True

def RoundPhaseShift(P):
    while P < 0:
        P += 2 * math.pi
    return float(P % (2 * math.pi))

def SignalComparePhaseShift(SignalInput, SignalOutput):
    if len(SignalInput) != len(SignalOutput):
        return False
    else:
        for i in range(len(SignalInput)):
            A = RoundPhaseShift(SignalInput[i])
            B = RoundPhaseShift(SignalOutput[i])
            if abs(A - B) > 0.0001:
                return False
    return True
    
def main():
    st.title("Fourier Transform App")

    # Input signals
    st.subheader("Input Signals")

    num_signals = st.number_input("How many signals do you want to analyze?", min_value=1, value=1, key="num_signals")

    input_signals = []
    output_x, output_y = [], []

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
            output_x_values, output_y_values = parse_signal_data(lines)
    
            # Ensure that the values are correctly converted to float when parsing the data
            output_x.extend(float(x) for x in output_x_values)
            output_y.extend(float(y) for y in output_y_values)

    if st.button('Perform Fourier Transform'):
        magnitude, phase = dft(y_values)
        output_x_array = np.array(output_x, dtype=float)
        output_y_array = np.array(output_y, dtype=float)
        # Print the values of amplitude and phase
        st.subheader("Amplitude Values:")
        st.write(magnitude)

        st.subheader("Phase Values:")
        st.write(phase)

        st.subheader("x Values:")
        st.write(output_x_array)

        st.subheader("y Values:")
        st.write(output_y_array)
       
        if (SignalCompareAmplitude(output_x_array, magnitude)):
            st.success("Amplitude values are correct.")
        else:
            st.error("Amplitude values are wrong.")

        if (SignalComparePhaseShift(output_y_array,phase)):
            st.success("phase values are correct.")
        else:
            st.error("phase values are wrong.")
                    
if __name__ == "__main__":
    main()