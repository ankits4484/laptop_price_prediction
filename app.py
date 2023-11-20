import streamlit as st
import pickle
import numpy as np
import sklearn 
import pandas as pd

#import the model
pipe=pickle.load(open('pipe.pkl','rb'))
df=pickle.load(open('df.pkl','rb'))

import streamlit as st

def add_bg_from_url():
    st.markdown(
         f"""
         <style>
         .stApp {{
             background-image: url("https://images.unsplash.com/photo-1593642702821-c8da6771f0c6?q=80&w=1000&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxzZWFyY2h8M3x8bGFwdG9wc3xlbnwwfHwwfHx8MA%3D%3D");
             background-attachment: fixed;
             background-size: cover;
             

         }}
        

         .stHeadingContainer {{
            
             color: red;
             border: 7px solid rgb(204, 0, 0);
             margin: 2px;
             border-radius: 2px;
             text-align: center;
             text-shadow: 1px 1px;
             font-size: 60px;
             padding: 25px 0;
             
  

         }}
         
          .stVerticalBlock{{
             border: 2px solid red;
             border-radius: 5px;
             font-size: 50px;
             display: flex;
             justify-content: flex-start;
         }}
         </style>
         """,
         unsafe_allow_html=True
     )

add_bg_from_url() 




st.title('Laptop Price Predictor')
container = st.container()
with container:

    # df=pd.read_csv('laptop_data.csv')
    company=st.selectbox('Brand',df['Company'].unique())

    type =st.selectbox('Type of Laptop',df['TypeName'].unique())

    ram = st.selectbox('RAM(in GB)',[ 8, 16,  4,  2, 12,  6, 32, 24, 64])

   

    touchscreen = st.selectbox('Touchscreen',['No','Yes'])

    ips = st.selectbox('IPS',['No','Yes'])

    

    resolution = st.selectbox('Screen Resolution',['1920x1080','1366x768','1600x900','3840x2160','3200x1800','2880x1800','2560x1600','2560x1440','2304x1440'])

    cpu = st.selectbox('CPU',['Intel Core i5', 'Intel Core i7', 'AMD Processor', 'Intel Core i3',
            'Other Intel Processor'])

    hdd = st.selectbox('HDD(in GB)',[0,128,256,512,1024,2048])

    ssd = st.selectbox('SSD(in GB)',[0,8,128,256,512,1024])

    gpu = st.selectbox('GPU',['Intel', 'AMD', 'Nvidia'])  

    os = st.selectbox('OS',['Mac', 'Others/No OS/Linux', 'Windows'])
    weight = st.slider('Weight of laptop(in kg)', 0.5, 4.0, 1.0)
    screen_size = st.slider('Scrensize in inches', 10.0, 18.0, 13.0)


    st.write('<style>div.stButton > button:first-child {background-color: SlateBlue; color: white; border: none; border-radius: 5px; padding: 10px 20px; font-size: 16px; cursor: pointer; align: center;}</style>', unsafe_allow_html=True)


    if st.button('Predict Price'):
            # query
            ppi = None
            if touchscreen == 'Yes':
                touchscreen = 1
            else:
                touchscreen = 0

            if ips == 'Yes':
                ips = 1
            else:
                ips = 0

            X_res = int(resolution.split('x')[0])
            Y_res = int(resolution.split('x')[1])
            ppi = ((X_res**2) + (Y_res**2))**0.5/screen_size
            query = np.array([company,type,int(ram),weight,touchscreen,ips,ppi,cpu,hdd,ssd,gpu,os])

            query = query.reshape(1,12)
            print(query)
            print(int(np.exp(pipe.predict(query)[0])))
            st.title("The predicted price of this configuration is Rs " + str(int(np.exp(pipe.predict(query)[0]))))