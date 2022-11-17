import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import pickle

st.image("./pic/banner.jpg")

html_8="""
<div style="background-color:#EE9513;padding:15px;border-radius:15px 15px 15px 15px;border-style:'solid';border-color:black">
<center><h5>การทำนายข้อมูลดอกไม้</h5></center>
</div>
"""

st.markdown(html_8,unsafe_allow_html=True)
st.markdown("")

dt=pd.read_csv("./data/water_potability.csv")
st.write(dt.head(10))
data1 = dt['ph'].sum()
data2 = dt['Hardness'].sum()
data3 = dt['Solids'].sum()
data4 = dt['Chloramines'].sum()
data5 = dt['Sulfate'].sum()
data6 = dt['Conductivity'].sum()
data7 = dt['Organic_carbon'].sum()
data8 = dt['Trihalomethanes'].sum()
data9 = dt['Turbidity'].sum()
dx=[data1,data2,data3,data4,data5,data6,data7,data8,data9]
dx2=pd.DataFrame(dx, index=["d1", "d2", "d3", "d4", "d5", "d6", "d7", "d8", "d9"])
if st.button("แสดงการจินตทัศน์ข้อมูล"):
   st.area_chart(dx2)
   st.button("ไม่แสดงข้อมูล")
else:
    st.write("ไม่แสดงข้อมูล")

html_8="""
<div style="background-color:#EE9513;padding:15px;border-radius:15px 15px 15px 15px;border-style:'solid';border-color:black">
<center><h5>การทำนายข้อมูล</h5></center>
</div>
"""
st.markdown(html_8,unsafe_allow_html=True)
st.markdown("")

w_ph=st.number_input("กรุณาเลือกข้อมูล ph")
w_ha=st.number_input("กรุณาเลือกข้อมูล Hardness")
w_so=st.number_input("กรุณาเลือกข้อมูล Solids")
w_ch=st.number_input("กรุณาเลือกข้อมูล Chloramines")
w_s=st.number_input("กรุณาเลือกข้อมูล Sulfate")
w_c=st.number_input("กรุณาเลือกข้อมูล Conductivity")
w_or=st.slider("กรุณาเลือกข้อมูล Organic_carbon")
w_tri=st.slider("กรุณาเลือกข้อมูล Trihalomethanes")
w_tur=st.slider("กรุณาเลือกข้อมูล Turbidity")



if st.button("ทำนายผล"):
   loaded_model = pickle.load(open('./data/water_trained_model.sav', 'rb'))
   input_data =  (w_ph,w_ha,w_so,w_ch,w_s,w_c,w_Or,w_Tri,w_Tur)
   # changing the input_data to numpy array
   input_data_as_numpy_array = np.asarray(input_data)
    # reshape the array as we are predicting for one instance
   input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)
   prediction = loaded_model.predict(input_data_reshaped)
   st.write(prediction)
   if prediction == '1':
        st.write("ใช่ได้")
   else :
        st.write("ไม่ได้")

    