import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

def sum_signals(signals):
    if len(signals) == 0:
        return []

    # Calculate the sum of y-values for all signals
    result = np.sum(signals, axis=0)
    return result

def subtract_signals(signals):
    if len(signals) < 2:
        return []

    # Subtract the second signal from the first signal
    result = abs(signals[0] - np.sum(signals[1:], axis=0))
    return result

def multiply_signal(signal, constant):
    # Multiply each element in the signal list by the constant
    result = [value * constant for value in signal]
    return result


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

def main():
    st.title('Signal Operations')

    operation = st.selectbox('Operation', ['Addition', 'Subtraction', 'Multiplication'])

    if operation in ['Addition', 'Subtraction']:
        num_signals = st.number_input('Number of Signals', min_value=2, value=2, step=1)
    else:  # For multiplication
        num_signals = 1

    constant = 0

    if operation == 'Multiplication':
        constant = st.number_input('Constant for Multiplication', value=1.0)

    signals = []

    for i in range(num_signals):
        signal_file = st.file_uploader(f'Upload Signal', type=['txt'], key=f'signal_uploader_{i}')
        if signal_file is not None:
            uploaded_data = signal_file.read().decode('utf-8')
            lines = uploaded_data.split('\n')
            x_values, y_values = parse_signal_data(lines)
            signals.append((x_values, y_values))

    if st.button('Perform Operation'):
        if operation in ['Addition', 'Subtraction']:
            if operation == 'Addition':
                result_signal = sum_signals([signal[1] for signal in signals])
                result_operation = 'Addition'
            elif operation == 'Subtraction':
                result_signal = subtract_signals([signal[1] for signal in signals])
                result_operation = 'Subtraction'
        else:  # Multiplication
            result_signal = multiply_signal(signals[0][1], constant)
            result_operation = 'Multiplication'

        st.success(f'Signal {result_operation} (Samples):')
        st.code(format_samples(result_signal), language='text')

        st.subheader('Original Signal (Discrete)')
        for signal in signals:
            fig = create_matplotlib_plot(signal[0], signal[1], 'Original Signal (Discrete)', continuous=False)
            st.pyplot(fig)

        st.subheader('Original Signal (Continuous)')
        for signal in signals:
            fig = create_matplotlib_plot(signal[0], signal[1], 'Original Signal (Continuous)', continuous=True)
            st.pyplot(fig)

        st.subheader(f'{result_operation} of Signal (Discrete)')
        fig = create_matplotlib_plot(signals[0][0], result_signal, f'{result_operation} of Signal (Discrete)', continuous=False)
        st.pyplot(fig)

        st.subheader(f'{result_operation} of Signal (Continuous)')
        fig = create_matplotlib_plot(signals[0][0], result_signal, f'{result_operation} of Signal (Continuous)', continuous=True)
        st.pyplot(fig)

if __name__ == '__main__':
    main()