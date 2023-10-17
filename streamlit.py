import streamlit as st

#make header
st.header('inputs', divider=True)

# upload txt
file = st.file_uploader("", type=["txt"])

# choose
trig_func = st.radio('Trig function',['sin','cos'])


Analog_freq = st.number_input("Analog frequency",min_value=0,max_value=10000,value=5)

#show plot
st.pyplot(plt)