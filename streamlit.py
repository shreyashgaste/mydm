import streamlit as st
import pandas as pd
import numpy as np
st.title("ESE")
menu = st.sidebar.selectbox("Menu",['Mean','Graph'])
file = st.file_uploader("Enter Dataset first to Proceed", type=['csv'], accept_multiple_files=False, disabled=False)

def loadDataset():
    data = pd.read_csv(file)
    print(data)
    return data

if file:
    data = loadDataset()
    st.header("Dataset Table")
    st.dataframe(data, width=1000, height=500)
    st.set_option('deprecation.showPyplotGlobalUse', False)
    operation = st.selectbox("Operation", ["Measure Central Tendency",'Dispersion','Analytical Plots'], index=0)
    if menu == 'Mean':
        cols = data.columns    
        atr1, atr2 = st.columns(2)
        attribute1 = atr1.selectbox("Select Attribute 1", cols)
        attribute2 = atr2.selectbox("Select Attribute 2", cols)
        def Mean():
            sum1 = np.sum(np.array(data.loc[:,attribute1]))
            avg1 = sum1/len(data)
            sum2 = np.sum(np.array(data.loc[:,attribute2]))
            avg2 = sum2/len(data)
            st.markdown("Mean " + attribute1 + ": " + str(avg1))
            st.write("===================================================")
            st.write("Mean " + attribute2 + ": " + str(avg2))
        if attribute1 and attribute2:
            Mean()
    if menu == 'Graph':
        print('graph')
    