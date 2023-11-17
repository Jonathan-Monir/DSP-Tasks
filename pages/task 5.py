import streamlit as st
if "input_signals" in st.session_state:
    input_signals = st.session_state["input_signals"]
    
if "output_signals" in st.session_state["output_signals"]:
    output_signals = st.session_state["output_signals"]
import math

def dct(input_list):
    N = len(input_list)
    result = []

    for k in range(N):
        sum_result = 0
        for n in range(N):
            sum_result += input_list[n] * math.cos(math.pi / (4 * N) * (2 * n - 1) * (2 * (k-1) + 1))
            
        result.append(math.sqrt(2 / N) * sum_result)
        
    return result

# Example usage

x_values, y_values, z_values = input_signals[0]
dct_result = dct(y_values)
st.write("Input values:", y_values)
st.write("DCT result:", dct_result)
