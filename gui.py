# import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import math
import numpy as np
import streamlit as st
import pandas as pd
import plotly.subplots as sp

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


# Header
st.header('inputs', anchor=None, help=None, divider=True)

# inputs
trig_func = st.radio('Trig function',['sin','cos'])
amp = st.number_input("Amplitude",min_value=1,max_value=1000,value=10)
Analog_freq = st.number_input("Analog frequency",min_value=0,max_value=10000,value=5)
Sample_freq = st.number_input("Sample frequency",min_value=1,max_value=10000,value=20)
phase_shift = st.number_input("Phase shift",min_value=float(0),max_value=float(100),step=0.2)
x = range(st.number_input("number of samples",min_value=1,max_value=10000,value=Sample_freq))
Fs = (Analog_freq/Sample_freq)

# Check that chosen sampling frequency is below the minumum required fs
fs_min = 2 * Analog_freq

if Sample_freq < fs_min:
    raise Exception(f"Warning: The chosen sampling frequency ({Sample_freq} Hz) is below the minimum required ({fs_min} Hz).")

Discrete_function = np.vectorize(Discrete_signal_generator)
y_input = Discrete_function(trig_func, amp, Fs, phase_shift, x)

st.header('plot via input', anchor=None, help=None, divider=True)
fig1 = st.plotly_chart(Discrete_plot(x, y_input, "Discrete Signal"))
fig2 = st.plotly_chart(Continuous_plot(x, y_input, "Continuous Signal"))
