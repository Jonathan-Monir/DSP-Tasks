import math
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

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
if "input_signals" in st.session_state:
    input_signals = st.session_state["input_signals"]
    
else:
    st.warning("Please add input signals")
    
if "output_signals" in st.session_state:
    output_signals = st.session_state["output_signals"]

else:
    st.warning("Please add output signals")

operation = st.selectbox("choose the function",["dct","remove dc"])

if operation == "remove dc":
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

    def RemoveDc(input_signals):
        # Extract y_values from each tuple
        y_values_list = [signal[1] for signal in input_signals]

        # Calculate the sum of y_values across all signals
        total_sum = sum([sum(y_values) for y_values in y_values_list])

        # Calculate the average
        average = total_sum / sum(len(y_values) for y_values in y_values_list)

        # Subtract the average from each y_values list
        result = [[x, [y - average for y in y_values]] for x, y_values in input_signals]

        return result
    def SignalSamplesAreEqual(file_name,samples):
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
                    
        if len(expected_samples)!=len(samples):
            st.write(len(expected_samples))
            st.write(len(samples))
            st.write("Test case failed, your signal have different length from the expected one")
            return
        for i in range(len(expected_samples)):
            if abs(samples[i] - expected_samples[i]) < 0.01:
                continue
            else:
                st.error("Test case failed, your signal have different values from the expected one") 
                return
        st.success("Test case passed successfully")
    
    def main():
        # Input signals
        st.subheader("Input Signals")

        num_signals = st.number_input("How many signals do you want to analyze?", min_value=1, value=1, key="num_signals")

        input_signals = []

        for i in range(num_signals):
            signal_file = st.file_uploader(f'Upload Signal {i + 1}', type=['txt'], key=f'signal_uploader_{i}')

            if signal_file is not None:
                uploaded_data = signal_file.read().decode('utf-8')
                lines = uploaded_data.split('\n')
                x_values, y_values = parse_signal_data(lines)
                input_signals.append((x_values, y_values))

        remove_dc_button = st.button('Remove DC Component')

        if remove_dc_button:
            result = RemoveDc(input_signals)
            st.subheader("Result after Removing DC Component")
            st.text("X Values:")
            st.text(result[0][0])
            st.text("Y Values:")
            st.text(result[0][1])
            st.plotly_chart(Discrete_plot(result[0][0], result[0][1], "Signal After DC Removal"))

            # Extract the y values from result
            y_values = result[0][1]
            
            # Move SignalSamplesAreEqual outside the button block
            file_name = r"files\task 5\Remove DC component\DC_component_output.txt"
            SignalSamplesAreEqual(file_name, y_values)

    if __name__ == "__main__":
        main()

######################################################################################################
else:
    def SignalSamplesAreEqual(file_name,samples):
        """
        this function takes two inputs the file that has the expected results and your results.
        file_name : this parameter corresponds to the file path that has the expected output
        samples: this parameter corresponds to your results
        return: this function returns Test case passed successfully if your results is similar to the expected output.
        """
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
        if len(expected_samples)!=len(samples):
            st.write("Test case failed, your signal have different length from the expected one")
            return
        for i in range(len(expected_samples)):
            if abs(samples[i] - expected_samples[i]) < 0.01:
                continue
            else:
                st.error("Test case failed, your signal have different values from the expected one") 
                return
        st.success("Test case passed successfully")

    def dct(input_list):
        
        result = []

        for k in range(N):
            sum_result = 0
            for n in range(N):
                sum_result += input_list[n] * math.cos(math.pi / (4 * N) * (2 * n - 1) * (2 * k - 1))
                
            result.append(math.sqrt(2 / N) * sum_result)
            
        return result


    # Example usage

    x_values, y_values, z_values = input_signals[0]
    N = len(y_values)
    dct_result = dct(y_values)
    number_of_coefficients = st.number_input("choose the first number of coefficients",1,N,value=N)

    st.table(dct_result[0:number_of_coefficients])
    st.plotly_chart(Discrete_plot(x_values,dct_result,'dct plot'))
    
    SignalSamplesAreEqual(r"files\task 5\DCT\DCT_output.txt",dct_result)
    if st.button("Download txt file"):
        with open('output_file', 'w') as file:
            for i in range(number_of_coefficients):
                file.write(f'{dct_result[i]:.6f}\n')
