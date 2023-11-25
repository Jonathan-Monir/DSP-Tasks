import streamlit as st

if "input_signals" in st.session_state:
    input_signals = st.session_state["input_signals"]
    
else:
    st.warning("Please add input signals")
    
if "output_signals" in st.session_state:
    output_signals = st.session_state["output_signals"]
    
else:
    st.warning("Please add output signals")
    
st.write("input signals: ", input_signals,"output signals:", output_signals)