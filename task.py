import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import pandas as pd

# Quantization function
def quantize_signal(input_signal, num_levels):
    # Ensure input_signal is a NumPy array
    input_signal = np.array(input_signal)
    
    max_val = max(input_signal)
    min_val = min(input_signal)
    step_size = (max_val - min_val) / (num_levels - 1)
    
    # Perform the quantization
    quantized_signal = np.round((input_signal - min_val) / step_size) * step_size + min_val
    
    return quantized_signal


# Function to create a Matplotlib plot
def create_matplotlib_plot(x, y, plot_name, continuous=True):
    fig, ax = plt.subplots()
    if continuous:
        ax.plot(x, y)
    else:
        ax.stem(x, y, linefmt='-b', markerfmt='ob', basefmt=' ')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title(plot_name)
    return fig

# Function to create a Plotly plot
def create_plotly_plot(x, y, plot_name):
    df = pd.DataFrame({'x': x, 'y': y})
    fig = px.line(df, x='x', y='y', title=plot_name)
    return fig

# Function to perform the quantization test
def QuantizationTest1(test_signal, quantized_signal):
    if len(test_signal) != len(quantized_signal):
        st.error("QuantizationTest1 Test case failed, the signals have different lengths.")
        return

    for i in range(len(test_signal)):
        if not np.isclose(quantized_signal[i], test_signal[i], atol=0.01):
            st.error("QuantizationTest1 Test case failed, the quantized signal has different values from the test signal.")
            return

    st.success("QuantizationTest1 Test case passed successfully")

def main():
    st.title('Signal Quantization')

    original_signal = []
    test_signal = []

    original_signal_file = st.file_uploader('Upload Original Signal', type=['txt'], key='original_signal_uploader')
    if original_signal_file is not None:
        uploaded_data = original_signal_file.read().decode('utf-8')
        lines = uploaded_data.split('\n')
        x_values, y_values = [], []
        for line in lines:
            parts = line.strip().split()
            if len(parts) == 2:
                x, y = parts
                try:
                    x_values.append(float(x))
                    y_values.append(float(y))
                except ValueError:
                    pass
        original_signal.append((x_values, y_values))

    test_signal_file = st.file_uploader('Upload Test Signal', type=['txt'], key='test_signal_uploader')
    if test_signal_file is not None:
        uploaded_data = test_signal_file.read().decode('utf-8')
        lines = uploaded_data.split('\n')
        test_x_values, test_y_values = [], []
        for line in lines:
            parts = line.strip().split()
            if len(parts) == 2:
                x, y = parts
                try:
                    test_x_values.append(float(x))
                    test_y_values.append(float(y))
                except ValueError:
                    pass
        test_signal.append((test_x_values, test_y_values))

    num_bits_or_levels = st.number_input("Enter the number of bits or levels:", min_value=1, key="num_bits_or_levels")

    if st.button('Perform Quantization'):
        if original_signal and test_signal:
            original_x_values, original_y_values = original_signal[0]
            test_x_values, test_y_values = test_signal[0]

            if num_bits_or_levels >= 2:
                num_levels = 2 ** num_bits_or_levels  # Calculate the number of levels from the number of bits
            else:
                num_levels = int(num_bits_or_levels)

            # Quantize the original signal
            quantized_signal = quantize_signal(original_y_values, num_levels)

            # Compute the quantization error
            quantization_error = original_y_values - quantized_signal

            # Display the original and quantized signals using Matplotlib
            st.subheader('Original Signal (Matplotlib)')
            st.pyplot(create_matplotlib_plot(original_x_values, original_y_values, 'Original Signal'))

            st.subheader('Quantized Signal (Matplotlib)')
            st.pyplot(create_matplotlib_plot(original_x_values, quantized_signal, 'Quantized Signal'))

            # Display the quantization error using Matplotlib
            st.subheader('Quantization Error (Matplotlib)')
            st.pyplot(create_matplotlib_plot(original_x_values, quantization_error, 'Quantization Error'))

            # Display the encoded signal
            st.subheader('Encoded Signal')
            st.write(quantized_signal)

            # Perform the quantization test using the provided test signal
            QuantizationTest1(test_y_values, quantized_signal)

if __name__ == '__main__':
    main()
