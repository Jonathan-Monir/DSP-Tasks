import streamlit as st
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from scipy.fft import fft
from scipy.fft import ifft
import math
import matplotlib.pyplot as plt
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

def format_samples(signal):
    return "\n".join([f"{x} {y}" for x, y in enumerate(signal)])

def parse_signal_data(data):
    x_values, y_values = [], []
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


# Function for making Discrete plot
def Discrete_plot(x, y, plot_name):
    fig = px.scatter(x=x, y=y, title=plot_name)

    for i in range(len(x)):
        fig.add_trace(go.Scatter(x=[x[i], x[i]], y=[0, y[i], None], mode='lines', name=f'Line {i}', line=dict(color='gray', dash='dash')))

    fig.update_layout(
        xaxis_title='n',
        yaxis_title='X(n)',
        showlegend=True
    )
    
    return fig
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
            x_list = [x for x in output_x_values]
            y_list = [y for y in output_y_values]
            output_x.extend(x_list)
            output_y.extend(y_list)
    option  = st.selectbox("Select the option",["dft","idft"])
    fs = st.number_input("sampling frequency",0,10000,1)
    add_amp = st.number_input("Adding amplitude",0,10000,0)
    multiply_amp = st.number_input("Multiply amplitude",0,10000,1)
    shift_phase = st.number_input("Shifting phase",-10000,10000,0)
    if st.button('Perform Fourier Transform'):
        if option == "dft":
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
            
            magnitude = ((magnitude * multiply_amp) + add_amp) / fs
            phase = (phase - shift_phase) / fs
            
            st.subheader("Amplitude Values after sample space:")
            st.write(magnitude)
            
            st.subheader("Phase Values after sample space:")
            st.write(phase)
            
            if (SignalCompareAmplitude(output_x_array, magnitude)):
                st.success("Amplitude values are correct.")
            else:
                st.error("Amplitude values are wrong.")

            if (SignalComparePhaseShift(output_y_array,phase)):
                st.success("phase values are correct.")
            else:
                st.error("phase values are wrong.")
            
            amps = [pow((magnitude[i]**2)+(phase[i]**2),1/2) for i in range(len(magnitude))]
            
            fundamental_frequency = [(2 * math.pi) / len(magnitude) * (1 / (fs * 0.001)) * amp for amp in range(len(magnitude))]
            
            phases = [math.degrees(math.atan(phase[i]/magnitude[i])) for i in range(len(magnitude))]
            
            st.plotly_chart(Discrete_plot(fundamental_frequency, amps, "Frequency versus Amplitude"))
            st.plotly_chart(Discrete_plot(fundamental_frequency, phases, "Frequency versus phase"))
            st.plotly_chart(Discrete_plot(amps, phases, "Magnitude versus phase"))
            
        if option == "idft":
            magnitude, phase = x_values, y_values
            
            
            st.write(magnitude,phase)
            output_y_array = np.array(output_y, dtype=float)
            
            magnitude = np.real(idft(magnitude, phase))
            
            magnitude = [round(((x * multiply_amp) + add_amp) * fs) for x in magnitude]
            
            st.subheader("Amplitude Values:")
            st.write(magnitude)

            if (SignalCompareAmplitude(output_y_array, magnitude)):
                st.success("Amplitude values are correct.")
            else:
                st.error("Amplitude values are wrong.")
        
        with open('data.txt', 'w') as file:
            for i, j in zip(magnitude, phase):
                file.write(f"{i} {j}\n")

        # Streamlit app
        st.title("Download Data as Text File")

        # Create a download link for the text file
        st.markdown("[Download Data](data.txt)")
        # Display frequency versus amplitude
        # st.subheader("Frequency versus Amplitude")
        # plt.stem(magnitude, use_line_collection=True)
        # st.pyplot()

        # Display frequency versus phase
        # st.subheader("Frequency versus Phase")
        # plt.stem(fs, phase, use_line_collection=True)
        # st.pyplot()

        # Display the magnitude and phase values
        # st.subheader("Magnitude and Phase Values")
        for i in range(len(magnitude)):
            st.write(f"Magnitude: {magnitude[i]}, Phase: {phase[i]}")
if __name__ == "__main__":
    main()