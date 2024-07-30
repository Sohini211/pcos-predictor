import pickle

import numpy as np
import streamlit as st
import pandas as pd
# import the model
pipe = pickle.load(open('PIPE (2).pkl','rb'))
df = pickle.load(open('DF (5).pkl','rb'))
st.title("PCOS Diagnosis Assistant")

# Input features from the user
st.sidebar.header('User Input Features')


def user_input_features():
    weight = st.number_input("Weight (Kg)", min_value=0.0, step=0.1)
    cycle = st.selectbox("Cycle(R/I)",["R" ,"I"])
    marriage_status = st.number_input("Marriage Status (Yrs)", min_value=0, step=1)
    beta_hcg1 = st.number_input("I beta-HCG(mIU/mL)", min_value=0.0, step=0.1)
    beta_hcg2 = st.number_input("II beta-HCG(mIU/mL)", min_value=0.0, step=0.1)
    fsh_lh = st.number_input("FSH/LH", min_value=0.0, step=0.1)
    amh = st.number_input("AMH(ng/mL)", min_value=0.0, step=0.1)
    vit_d3 = st.number_input("Vit D3 (ng/mL)", min_value=0.0, step=0.1)
    weight_gain = st.selectbox("Weight gain(Y/N)", ["Y", "N"])
    hair_growth = st.selectbox("Hair growth(Y/N)", ["Y", "N"])
    skin_darkening = st.selectbox("Skin darkening (Y/N)", ["Y", "N"])
    pimples = st.selectbox("Pimples(Y/N)", ["Y", "N"])
    fast_food = st.selectbox("Fast food (Y/N)", ["Y", "N"])
    follicle_no_l = st.number_input("Follicle No. (L)", min_value=0, step=1)
    follicle_no_r = st.number_input("Follicle No. (R)", min_value=0, step=1)




    data = {
        'Weight (Kg)': weight,
        'Cycle(R/I)': cycle,
        'Marraige Status (Yrs)': marriage_status,
        'I   beta-HCG(mIU/mL)': beta_hcg1,
        'II    beta-HCG(mIU/mL)': beta_hcg2,
        'FSH/LH': fsh_lh,
        'AMH(ng/mL)': amh,
        'Vit D3 (ng/mL)': vit_d3,
        'Weight gain(Y/N)': weight_gain,
        'hair growth(Y/N)': hair_growth,
        'Skin darkening (Y/N)': skin_darkening,
        'Pimples(Y/N)': pimples,
        'Fast food (Y/N)': fast_food,
        'Follicle No. (L)': follicle_no_l,
        'Follicle No. (R)': follicle_no_r,
    }


    features = pd.DataFrame(data, index=[0])
    return features

df = user_input_features()

# Display user input
st.write('User Input Features:')
st.write(df)



# Convert categorical inputs to numerical values if necessary


df['Cycle(R/I)'] = df['Cycle(R/I)'].map({'R': 2, 'I': 4})
df['Weight gain(Y/N)'] = df['Weight gain(Y/N)'].map({'Yes': 1, 'No': 0})
df['hair growth(Y/N)'] = df['hair growth(Y/N)'].map({'Yes': 1, 'No': 0})
df['Skin darkening (Y/N)'] = df['Skin darkening (Y/N)'].map({'Yes': 1, 'No': 0})
df['Pimples(Y/N)'] = df['Pimples(Y/N)'].map({'Yes': 1, 'No': 0})
df['Fast food (Y/N)'] = df['Fast food (Y/N)'].map({'Yes': 1, 'No': 0})

df = df.astype(float)
# Make prediction
prediction = pipe.predict(df)
prediction_proba = pipe.predict_proba(df)
if st.button('Predict'):
# Display the prediction
    st.write('Prediction:', 'PCOS' if prediction[0] == 1 else 'No PCOS')
    st.write('Prediction Probability:', prediction_proba)