import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

def sum_signals(signals):
    if len(signals) == 0:
        return []

    # Calculate the sum of y-values for all signals
    result = np.sum(signals, axis=0)
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
    plt.figure(figsize=(8, 4))
    if continuous:
        plt.plot(x, y)
    else:
        plt.stem(x, y, linefmt='-b', markerfmt='ob', basefmt=' ')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(plot_name)
    return plt

def main():
    st.title('Signal Summation and Plotting')

    num_signals = st.number_input('Number of Signals', min_value=2, max_value=10, value=2, step=1)
    signals = []

    for i in range(num_signals):
        signal_file = st.file_uploader(f'Upload Signal #{i+1}', type=['txt'])
        if signal_file is not None:
            uploaded_data = signal_file.read().decode('utf-8')
            lines = uploaded_data.split('\n')
            x_values, y_values = parse_signal_data(lines)
            signals.append((x_values, y_values))

    if st.button('Sum and Plot Signals'):
        summed_signal = sum_signals([signal[1] for signal in signals])

        st.success('Signal Sum (Samples):')
        st.code(format_samples(summed_signal), language='text')

        st.subheader('Original Signals (Discrete)')
        for i, (x, y) in enumerate(signals):
            create_matplotlib_plot(x, y, f'Signal {i+1} (Discrete)', continuous=False)
            st.pyplot()

        st.subheader('Original Signals (Continuous)')
        for i, (x, y) in enumerate(signals):
            create_matplotlib_plot(x, y, f'Signal {i+1} (Continuous)', continuous=True)
            st.pyplot()

        st.subheader('Sum of Signals (Discrete)')
        create_matplotlib_plot(summed_signal[0], summed_signal[1], 'Sum of Signals (Discrete)', continuous=False)
        st.pyplot()

if __name__ == '__main__':
    main()
