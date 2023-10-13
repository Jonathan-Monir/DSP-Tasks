# import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import math
import numpy as np
import streamlit as st
import pandas as pd

# Function for cleaning file
def clean_text(file_list):
    signals = list()
    for line in file_list:
        signals.append(line.strip())
            
    x = [item.split(' ')[0] for item in signals[3:]]
    y = [item.split(' ')[1] for item in signals[3:]]
    z = 0
    if signals[0] == '1':
        z = [item.split(' ')[2] for item in signals[3:]]
    return x, y, z

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


# Function for making Continuous plot
def Continuous_plot(x, y, plot_name):
    df = pd.DataFrame({'x': x, 'y': y})
    fig = px.line(df, x='x', y='y', title=plot_name)
    fig.update_traces(mode='lines')
    
    return fig

# Function for making y based on the function
def Discrete_signal_generator(trig_func, Amplitude, Fs, phase_shift, n):
    
    trig_func = getattr(math,trig_func)
    w = 2 * math.pi * (Fs)
    y = Amplitude * trig_func(w*n + phase_shift)
    
    return y 


    
st.header('inputs', anchor=None, help=None, divider=True)

file = st.file_uploader("", type=["txt"])
# inputs
trig_func = st.radio('Trig function',['sin','cos'])
x = range(st.number_input("number of samples",min_value=1,max_value=10000,value=200))
amp = st.number_input("Amplitude",min_value=1,max_value=1000,value=10)
Analog_freq = st.number_input("Analog frequency",min_value=0,max_value=10000,value=5)
Sample_freq = st.number_input("Sample frequency",min_value=1,max_value=10000,value=200)
phase_shift = st.number_input("Phase shift",min_value=float(0),max_value=float(100),step=0.2)
Fs = (Analog_freq/Sample_freq)
# Check that chosen sampling frequency is below the minumum required fs
fs_min = 2 * Analog_freq

if Sample_freq < fs_min:
    raise Exception(f"Warning: The chosen sampling frequency ({Sample_freq} Hz) is below the minimum required ({fs_min} Hz).")
    
vectorized_function = np.vectorize(Discrete_signal_generator)
y_input = vectorized_function(trig_func, amp, Fs, phase_shift, x)

st.header('plot via input', anchor=None, help=None, divider=True)
st.plotly_chart(Discrete_plot(x, y_input, "Discrete Signal"))
st.plotly_chart(Continuous_plot(x, y_input, "Continuous Signal"))


# file
if file is not None:
    st.header('plot via File', anchor=None, help=None, divider=True)
    file_contents_bytes = file.read()

    file_contents_str = file_contents_bytes.decode('utf-8')
    lines = file_contents_str.split('\n')

    lines = [line.strip() for line in lines if line.strip()]
    
    x = clean_text(lines)[0]
    y = clean_text(lines)[1]
    
    if lines[0] == '1':
        trig_func = st.radio('Trig function',['sin','cos'],key=1)
        x = np.arange(int(lines[2]))
        
        Fs = float(clean_text(lines)[0][0])
        amp = float(clean_text(lines)[1][0])
        Ps = float(clean_text(lines)[2][0])
        
        if lines[1] == '1':
            Fs = 1/Fs
            
        vectorized_function = np.vectorize(Discrete_signal_generator)
        y = vectorized_function(trig_func, amp, Fs, Ps, x)
            
    st.plotly_chart(Discrete_plot(x,y,'Discrete Signal'))
    st.plotly_chart(Continuous_plot(x,y,'Continuous Signal'))