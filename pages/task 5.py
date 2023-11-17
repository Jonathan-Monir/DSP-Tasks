import math
import streamlit as st


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
            sum_result += input_list[n] * math.cos(math.pi / (4 * N) * (2 * n - 1) * (2 * (k - 1) + 1))
            
        result.append(math.sqrt(2 / N) * sum_result)
        
    return result


if "input_signals" in st.session_state:
    input_signals = st.session_state["input_signals"]
    
else:
    st.warning("Please add input signals")
    
if "output_signals" in st.session_state:
    output_signals = st.session_state["output_signals"]

else:
    st.warning("Please add output signals")

# Example usage

x_values, y_values, z_values = input_signals[0]
N = len(y_values)
dct_result = dct(y_values)
number_of_coefficients = st.number_input("choose the first number of coefficients",1,N,value=N)

st.table(dct_result[0:number_of_coefficients])

SignalSamplesAreEqual(r"files\task 5\DCT\DCT_output.txt",dct_result)
if st.button("Download txt file"):
    with open('output_file', 'w') as file:
        for i in range(number_of_coefficients):
            file.write(f'{dct_result[i]:.6f}\n')
