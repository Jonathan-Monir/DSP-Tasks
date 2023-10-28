import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import pandas as pd

def generate_binary_numbers(level):
    binary_numbers = []
    num_bits = level

    for i in range(level):
        binary_str = format(i,str(level.bit_length()) + 'b')
        binary_numbers.append(binary_str)

    return binary_numbers

# Quantization function
def quantize_signal(input_signal, num_levels):
    # Ensure input_signal is a NumPy array
    input_signal = np.array(input_signal)
    
    max_val = max(input_signal)
    min_val = min(input_signal)
    step_size = (max_val - min_val) / (num_levels)
    
    midpoints =  np.array([((min_val + (level * step_size) + (min_val + ((level+1) * step_size)))/2) for level in range(num_levels)])
    
    quantized_signal = []
    
    binary_numbers = generate_binary_numbers(num_levels)
    binary_mapping = {midpoints[i]: binary_numbers[i] for i in range(len(midpoints))}
    # Iterate through elements in the first list
    for elem1 in input_signal:
        # Find the nearest element in the second list using the min() function
        nearest_elem2 = min(midpoints, key=lambda x: abs(x - elem1))
        
        # Append the nearest element to the list
        quantized_signal.append(nearest_elem2)
        
    quantized_signal = np.array(quantized_signal)
    
    return quantized_signal, binary_mapping

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

    two_inputs = True
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
        test_x_values, test_y_values, test_z_values, test_W_values = [], [], [], []
        for line in lines:
            parts = line.strip().split()
            if len(parts) == 2:
                x, y = parts
                try:
                    test_x_values.append(float(x))
                    test_y_values.append(float(y))
                except ValueError:
                    pass
            if len(parts) == 4:
                two_inputs = False
                
                x, y, z, W = parts
                try:
                    test_x_values.append(float(x))
                    test_y_values.append(float(y))
                    test_z_values.append(float(z))
                    test_W_values.append(float(W))
                except ValueError:
                    pass
        test_signal.append((test_x_values, test_y_values, test_z_values, test_W_values))

    quantize_type = st.selectbox("",["bits","levels"])
    num_bits_or_levels = st.number_input(f"Enter the number of {quantize_type}:", min_value=1, key="num_bits_or_levels")
    if quantize_type == "bits":
        num_bits_or_levels = 2**num_bits_or_levels
    
    
    if st.button('Perform Quantization'):
        if original_signal and test_signal:
            original_x_values, original_y_values = original_signal[0]
            test_x_values, test_y_values, test_z_values, test_W_values = test_signal[0]

            # Quantize the original signal
            quantized_signal = quantize_signal(original_y_values, num_bits_or_levels)[0]
            binary_mapping = quantize_signal(original_y_values, num_bits_or_levels)[1]

            # Compute the quantization error
            quantization_error = quantized_signal - original_y_values
            cnt = 0
            for key in quantized_signal:
                if key in binary_mapping:
                    st.write(f"{binary_mapping[key]} -> {round(key,3)} -> {round(quantization_error[cnt],3)}")
                    cnt += 1
                else:
                    st.write(f"{key} not found in binary_mapping")

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
            
            if two_inputs:
                test_z_values = test_y_values
            # Perform the quantization test using the provided test signal
            QuantizationTest1(test_z_values, quantized_signal)

if __name__ == '__main__':
    main()
