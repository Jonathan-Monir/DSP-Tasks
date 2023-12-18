# import streamlit as st
import numpy as np
import math

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

def window_type(stopband_attenuation):
    if stopband_attenuation < 21:
        return 'rectangular', 0.9
    elif stopband_attenuation < 44:
         return 'hanning', 3.1
    elif stopband_attenuation < 53:
        return 'hamming', 3.3
    else:
         return 'blackman', 5.5

def low_pass(n, fc, fs, tw):
    new_fc = (fc + (tw/2))/fs
    if n == 0:
        return 2 * new_fc
    else:
        return 2 * new_fc * np.sin(2 * np.pi * new_fc * n) / (2 * np.pi * new_fc * n)

def high_pass(n, fc, fs, tw):
    new_fc = (fc - (tw/2))/fs
    if n == 0:
        return 1 - (2 * new_fc)
    else:
        return -2 * new_fc * np.sin(2 * np.pi * new_fc * n) / (2 * np.pi * new_fc * n)

def band_pass(n, fc_low, fc_high, fs, tw):
    return low_pass(n, fc_high, fs, tw) + high_pass(n, fc_low, fs, tw)

def band_reject(n, fc_low, fc_high, fs, tw):
    return (low_pass(n, fc_low, fs, tw) + high_pass(n, fc_high, fs, tw))



def rectangle(n, N):
    return 1

def hanning(n, N):
    return 0.5 + 0.5 * np.cos((2 * np.pi * n) / N)

def hamming(n, N):
    return 0.54 + 0.46 * np.cos((2 * np.pi * n) / N)

def blackman(n, N):
    return 0.42 + 0.5 * np.cos((2 * np.pi * n) / (N - 1)) + 0.08 * np.cos((4 * np.pi * n) / (N - 1))

def design_fir_filter(filter_type, fs, stopband_attenuation, fc1, fc2, transition_band):
    result = []
    delta_f = transition_band / fs
    window, ntw = window_type(stopband_attenuation)
    N = math.ceil(ntw / delta_f)
    if N % 2 == 0:
        N = N + 1

    start_value = int(-N/2)
    step = 1
    for i in range(start_value, int(N/2)+1, int(step)):
        w, h = None, None

        if window == "rectangular":
            w = rectangle(i, N)
        elif window == "hamming":
            w = hamming(i, N)
        elif window == "hanning":
            w = hanning(i, N)
        else:
            w = blackman(i, N)

        if filter_type == "lowpass":
            h = low_pass(i, fc1, fs, transition_band)
        elif filter_type == "highpass":
            h = high_pass(i, fc1, fs, transition_band)
        elif filter_type == "bandpass":
            h = band_pass(i, fc1, fc2, fs, transition_band)
        else:
            h = band_reject(i, fc1, fc2, fs, transition_band)

        result.append((i, h * w))

    return result ,start_value

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
    if (len(expected_samples) != len(Your_samples)) and (len(expected_indices) != len(Your_indices)):
        st.write(len(expected_samples), len(Your_samples))
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


def main():
    st.title("FIR Filter Design with Streamlit")
    filter_type = st.selectbox("Select filter type", ["lowpass", "highpass", "bandpass", "bandreject"])
    fs = st.number_input("Enter sampling frequency (FS): ")
    stopband_attenuation = st.number_input("Enter stopband attenuation: ")
    fc1 = st.number_input("Enter cutoff frequency (FC1): ")
    fc2 = 0
    if filter_type == "bandreject" or filter_type == "bandpass":
        fc2 = st.number_input("Enter cutoff frequency (FC2): ")
    transition_band = st.number_input("Enter transition band width: ")

    fir_coefficients, start_value = design_fir_filter(filter_type, fs, stopband_attenuation, fc1, fc2, transition_band)

    signal_file = st.file_uploader(f'Upload Signal {1}', type=['txt'], key=f'signal_uploader_{0}')
    

    if st.button("Show Filter Coefficients"):
        st.line_chart(fir_coefficients, use_container_width=True)
        for i, value in fir_coefficients:
            st.write(f"{i} {value:.8f}")

    input_signals = []

    if st.button("Apply Filter"):
        if signal_file is not None:
            uploaded_data = signal_file.read().decode('utf-8')
            lines = uploaded_data.split('\n')
            x_values, y_values = parse_signal_data(lines)
            input_signals.append((x_values, y_values))
            indices = []
            samples = []
            if fir_coefficients:
                convolved_signal = convolve(input_signals[0][1], [val for _, val in fir_coefficients], start_value)

                st.write("Convolved Signal:")
                st.write(len(convolved_signal[1]))  # Print the length of the convolved signal
                for i, value in zip(convolved_signal[0], convolved_signal[1]):
                    st.write(f"{i} {value:.8f}")  # Print index and value with 8 decimal places
                    indices.append(i)
                    samples.append(value)
                print(samples)
                Compare_Signals(r"files\FIR test cases\Testcase 6\ecg_band_pass_filtered.txt",indices,samples)
            else:
                st.warning("Please design the filter first before applying.")

if __name__ == "__main__":
    main()
