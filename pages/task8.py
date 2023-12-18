import streamlit as st
import math
import numpy as np
if "input_signals" in st.session_state:
    input_signals = st.session_state["input_signals"]
    
else:
    st.warning("Please add input signals")
    
if "output_signals" in st.session_state:
    output_signals = st.session_state["output_signals"]

else:
    st.warning("Please add output signals")

if "input_signals" in st.session_state:
    signal1 = input_signals[0][1]


operation = st.selectbox("select operation",["Fast convolution","Fast correlation"])


###############################################################################

if operation == "Fast convolution":
    import streamlit as st
    import numpy as np
    from scipy.fft import fft
    from scipy.fft import ifft


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
                    pass
        return x_values, y_values

    def dft(input_signal):
        fft_result = fft(input_signal)
        magnitude = np.abs(fft_result)
        phase = np.angle(fft_result)
        return magnitude, phase

    def idft(amplitude, phase):
        # Combine amplitude and phase to create a complex signal
        complex_signal = amplitude * np.exp(1j * np.array(phase))

        # Perform the IFFT
        result = ifft(complex_signal)
        return result

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

    def ConvTest(Your_indices,Your_samples): 
        """
        Test inputs
        InputIndicesSignal1 =[-2, -1, 0, 1]
        InputSamplesSignal1 = [1, 2, 1, 1 ]
        
        InputIndicesSignal2=[0, 1, 2, 3, 4, 5 ]
        InputSamplesSignal2 = [ 1, -1, 0, 0, 1, 1 ]
        """
        
        expected_indices=[-2, -1, 0, 1, 2, 3, 4, 5, 6]
        expected_samples = [1, 1, -1, 0, 0, 3, 3, 2, 1 ]

        
        if (len(expected_samples)!=len(Your_samples)) and (len(expected_indices)!=len(Your_indices)):
            print("Conv Test case failed, your signal have different length from the expected one")
            return
        for i in range(len(Your_indices)):
            if(Your_indices[i]!=expected_indices[i]):
                print("Conv Test case failed, your signal have different indicies from the expected one") 
                return
        for i in range(len(expected_samples)):
            if abs(Your_samples[i] - expected_samples[i]) < 0.01:
                continue
            else:
                print("Conv Test case failed, your signal have different values from the expected one") 
                return
        print("Conv Test case passed successfully")
    def main():
        st.title("Convolution Operation")

        num_signals = st.number_input("How many signals do you want to convolve?", min_value=2, value=2, key="num_signals")

        input_signals = []

        for i in range(num_signals):
            signal_file = st.file_uploader(f'Upload Signal {i + 1}', type=['txt'], key=f'signal_uploader_{i}')

            if signal_file is not None:
                uploaded_data = signal_file.read().decode('utf-8')
                lines = uploaded_data.split('\n')
                x_values, y_values = parse_signal_data(lines)
                input_signals.append((x_values, y_values))

        if len(input_signals) == 2:
            signal1 = input_signals[0][1]
            signal2 = input_signals[1][1]
            convolved_signal = convolve_freq_domain(signal1, signal2)

            st.subheader("Convolved Signal:")
            # Print convolved signal in the specified format
            for x, y in enumerate(convolved_signal):
                st.write(f"{x} {y}")

            # Run ConvTest and display result
            ConvTest(list(range(len(convolved_signal))), convolved_signal)
            
            # get the magnitude only
            magnitude = np.abs((convolved_signal))
            st.subheader("Magnitude of convolved signal:")
            st.write(magnitude)
            
    main()
###############################################################################


elif operation == "Fast correlation":
    import streamlit as st
    import numpy as np
    from scipy.fft import fft
    from scipy.fft import ifft

    def dft(input_signal):
        N = len(input_signal)
        magnitude = np.zeros(N)
        phase = np.zeros(N)

        for k in range(N):
            real_part = 0.0
            imag_part = 0.0

            for n in range(N):
                angle = 2 * np.pi * k * n / N
                real_part += input_signal[n] * np.cos(angle)
                imag_part -= input_signal[n] * np.sin(angle)

            magnitude[k] = np.sqrt(real_part**2 + imag_part**2)
            phase[k] = np.arctan2(imag_part, real_part)

        return magnitude, phase
    
    
    def idft(amplitude, phase):
        N = len(amplitude)
        complex_signal = np.zeros(N, dtype=complex)

        for k in range(N):
            real_part = 0.0
            imag_part = 0.0

            for n in range(N):
                angle = 2 * np.pi * k * n / N
                real_part += amplitude[n] * np.cos(angle)
                imag_part += amplitude[n] * np.sin(angle)

            complex_signal[k] = real_part + 1j * imag_part

        # Perform the Inverse FFT
        result = np.fft.ifft(complex_signal)
        return result
    
    X = dft(signal1)
    Y = dft(signal1) 
    
    Y = list(Y)
    Y[1] = np.array(Y[1]) * -1
    Y = tuple(Y)
    
    mag1, phase1 = X
    mag2, phase2 = Y
    
    mag_result = mag1 * mag2
    phase_result = phase1 + phase2

    # Perform IDFT on the result to get the convolution in time domain
    convolution_result = idft(mag_result, phase_result)
    
    print(type(convolution_result[0]))
    
    # extract only the magnitude of convolution result where it's type is numpy.complex128
    convolution_result = np.abs(convolution_result)
    
    print(convolution_result)
    
    # Print convolved signal in the specified format
    for x, y in enumerate(convolution_result):
        st.write(f"{x} {y}")
    # convolved_signal = convolve_freq_domain(X, Y)
