import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve
from scipy.fft import fft

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

def fold_signal(signal):
    folded_signal = []
    for i in range(len(signal)):
        folded_signal.append((signal[i][0], len(signal) - i))
    return folded_signal


def delay_normal(signal, delay_amount):
    x_values, y_values = zip(*signal)
    
    # Convert x values to numpy array for easy manipulation
    x_values = np.array(x_values)
    
    # Delay only the x values
    delayed_x_values = x_values + float(delay_amount)
    
    # Combine the delayed x values with the original y values into a single signal
    delayed_signal = list(map(lambda x, y: f"{x} {y}", delayed_x_values, y_values))
    
    return delayed_signal

def delay_signal(signal, delay_amount):
    # Extract x and y values from the signal
    signal = fold_signal(signal)
    x_values, y_values = zip(*signal)
    
    # Convert x values to numpy array for easy manipulation
    x_values = np.array(x_values)
    
    # Delay only the x values
    delayed_x_values = x_values + float(delay_amount)
    
    # Combine the delayed x values with the original y values into a single signal
    delayed_signal = list(map(lambda x, y: f"{x} {y}", delayed_x_values, y_values))
    
    return delayed_signal

def DerivativeSignal():
    InputSignal=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100]
    expectedOutput_first = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    expectedOutput_second = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    """
    Write your Code here:
    Start
    """
  
    FirstDrev=[InputSignal[i+1] - InputSignal[i] for i in range(len(InputSignal)-1)]
    SecondDrev=[InputSignal[i+2]-2*InputSignal[i+1]+InputSignal[i] for i in range(len(InputSignal)-2)]
    
    st.write(f"FirstDrev: ",FirstDrev," secondDrev: ",SecondDrev)
    
    
    """
    End
    """
    
    """
    Testing your Code
    """
    if( (len(FirstDrev)!=len(expectedOutput_first)) or (len(SecondDrev)!=len(expectedOutput_second))):
        st.write("mismatch in length") 
        return
    first=second=True
    for i in range(len(expectedOutput_first)):
        if abs(FirstDrev[i] - expectedOutput_first[i]) < 0.01:
            continue
        else:
            first=False
            st.write("1st derivative wrong")
            return
    for i in range(len(expectedOutput_second)):
        if abs(SecondDrev[i] - expectedOutput_second[i]) < 0.01:
            continue
        else:
            second=False
            st.write("2nd derivative wrong") 
            return
    if(first and second):
        st.success("Derivative Test case passed successfully")
    else:
        st.write("Derivative Test case failed")
    return


def remove_dc_component(input_signal):
    fft_result = np.fft.fft(input_signal)
    fft_result[0] = 0
    filtered_signal = np.fft.ifft(fft_result)

    return filtered_signal.real  

def convolve_signals(signal1, signal2):
    if len(signal1) == 0 or len(signal2) == 0:
        raise ValueError("Input signals cannot be empty for convolution.")
    convolved_signal = convolve(signal1, signal2, mode='full')
    return convolved_signal

def plot_signal(x_values, original_signal, processed_x, processed_signal):
    st.subheader("Signal:")
    st.write("Original Signal:", format_samples(list(zip(x_values, original_signal))))
    st.write("Processed Signal:", format_samples(list(zip(processed_x, processed_signal))))

    plt.figure(figsize=(10, 5))
    plt.plot(x_values, original_signal, label='Original Signal')
    plt.plot(processed_x, processed_signal, label='Processed Signal')
    plt.title('Processed Signal')
    plt.legend()
    st.pyplot()

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
        st.write("Conv Test case failed, your signal have different length from the expected one")
        return
    for i in range(len(Your_indices)):
        if(Your_indices[i]!=expected_indices[i]):
            st.write("Conv Test case failed, your signal have different indicies from the expected one") 
            return
    for i in range(len(expected_samples)):
        if abs(Your_samples[i] - expected_samples[i]) < 0.01:
            continue
        else:
            st.write("Conv Test case failed, your signal have different values from the expected one") 
            return
    st.success("Conv Test case passed successfully")

def Shift_Fold_Signal(file_name,Your_indices,Your_samples):      
    expected_indices=[]
    expected_samples=[]
    with open(file_name, 'r') as f:
        line = f.readline()
        line = f.readline()
        line = f.readline()
        line = f.readline()
        while line:
            # process line
            L=line.strip()
            if len(L.split(' '))==2:
                L=line.split(' ')
                V1=int(L[0])
                V2=float(L[1])
                expected_indices.append(V1)
                expected_samples.append(V2)
                line = f.readline()
            else:
                break
    st.write("Current Output Test file is: ")
    st.write(file_name)
    st.write("\n")
    if (len(expected_samples)!=len(Your_samples)) and (len(expected_indices)!=len(Your_indices)):
        st.write("Shift_Fold_Signal Test case failed, your signal have different length from the expected one")
        return
    for i in range(len(Your_indices)):
        if(Your_indices[i]!=expected_indices[i]):
            st.write(Your_indices[i],expected_indices[i])
            st.write("Shift_Fold_Signal Test case failed, your signal have different indicies from the expected one") 
            return
    for i in range(len(expected_samples)):
        if abs(Your_samples[i] - expected_samples[i]) < 0.01:
            continue
        else:
            st.write("Shift_Fold_Signal Test case failed, your signal have different values from the expected one") 
            return
    st.success("Shift_Fold_Signal Test case passed successfully")

def main():
    operation = st.radio("Choose Operation", ["Fold", "Delay","Delay folded", "Convolve", "Remove DC", "Moving average", "Derivative"])

    num_signals = st.number_input("How many signals do you want to analyze?", min_value=1, value=1, key="num_signals")

    input_signals = []

    for i in range(num_signals):
        signal_file = st.file_uploader(f'Upload Signal {i + 1}', type=['txt'], key=f'signal_uploader_{i}')
        output_signal_file = st.file_uploader(f'Upload Output Signal {i + 1}', type=['txt'], key=f'output_signal_uploader_{i}')

        if signal_file is not None:
            uploaded_data = signal_file.read().decode('utf-8')
            lines = uploaded_data.split('\n')
            x_values, y_values = parse_signal_data(lines)
            input_signals.append((x_values, y_values))

    if operation == "Delay folded" or operation == "Delay":
        delay_amount = st.number_input(f"Enter delay amount for all signals", value=0)
        
    if operation == "Moving average":
        window_size = st.number_input('window size',1,6,3)
        
    if st.button("Apply Operation"):
        for i, (x_values, y_values) in enumerate(input_signals):
            if operation == "Fold":
                indices = []
                samples = []
                folded_signal = fold_signal(list(zip(x_values, y_values)))
                st.subheader("Folded Signal:")
                # Print the folded signal in the specified format
                for x, y in folded_signal:
                    st.write(f"{x} {y}")
                    indices.append(x)
                    samples.append(y)
                Shift_Fold_Signal(r"files\task 6\TestCases\Shifting and Folding\Output_fold.txt",indices,samples)
            elif operation == "Delay folded":
                indices = []
                samples = []
                delayed_signal = delay_signal(list(zip(x_values, y_values)), delay_amount)
                
                st.subheader("Delayed folded Signal:")
                # Print the delayed signal in the specified format
                for signal in delayed_signal:
                    st.write(signal)
                    parts = signal.split(" ")
                    indices.append(float(parts[0]))
                    samples.append(int(parts[1]))
                if delay_amount == 0:
                    Shift_Fold_Signal(r"files\task 6\TestCases\Shifting and Folding\Output_fold.txt",indices,samples)
                elif delay_amount == -500:
                    Shift_Fold_Signal(r"files\task 6\TestCases\Shifting and Folding\Output_ShiftFoldedby-500.txt",indices,samples)
                elif delay_amount == 500:
                    Shift_Fold_Signal(r"files\task 6\TestCases\Shifting and Folding\Output_ShifFoldedby500.txt",indices,samples)
            elif operation == "Delay":
                delayed_signal = delay_signal(list(zip(x_values, y_values)), delay_amount)
                st.subheader("Delayed Signal:")
                # Print the delayed signal in the specified format
                for signal in delayed_signal:
                    st.write(signal)

            elif operation == "Convolve":
                indices = []
                samples = []
                if i == 0:
                    st.warning("Upload a second signal for convolution.")
                    continue
                indices1 = input_signals[0][0]
                signal1 = input_signals[0][1]
                signal2 = y_values
                convolved_signal = convolve_signals(signal1, signal2)
                convolved_x = np.arange(len(convolved_signal))
                # Print convolved signal in the specified format
                for x, y in zip(convolved_x, convolved_signal):
                    st.write(f"{x+indices1[0]} {y}")
                    indices.append(x+indices1[0])
                    samples.append(y)
                ConvTest(indices,samples)
                
            elif operation == "Remove DC":
                dc_removed_signal = remove_dc_component(y_values)
                # Print DC removed signal in the specified format
                for x, y in zip(x_values, dc_removed_signal):
                    st.write(f"{x} {y}")
########################################################################################################
    
        if operation == "Moving average":
            signal1 = input_signals[0][1]
            output = [sum(signal1[i:i+window_size])/window_size for i in range(len(signal1) - (window_size-1))]
            for i in output:
                st.write(str(i))
        if operation == "Derivative":
            DerivativeSignal()
if __name__ == "__main__":
    main()
