import streamlit as st
import pandas as pd
import joblib

# Load trained model
model = joblib.load("titanic_model.pkl")

# Page config
st.set_page_config(page_title="Titanic Survival Predictor", layout="centered")

# Title
st.title("🚢 Titanic Survival Prediction")
st.write("Fill in the passenger details below:")

# USER INPUTS
age = st.slider("Age", 0, 80, 25)
fare = st.number_input("Fare", min_value=0.0, max_value=500.0, value=50.0)

sibsp = st.number_input("Siblings/Spouses aboard (sibsp)", 0, 10, 0)
parch = st.number_input("Parents/Children aboard (parch)", 0, 10, 0)

sex = st.selectbox("Sex", ["male", "female"])
embarked = st.selectbox("Port of Embarkation", ["S", "C", "Q"])
pclass = st.selectbox("Passenger Class", ["First", "Second", "Third"])

who = st.selectbox("Who", ["man", "woman", "child"])
adult_male = st.selectbox("Adult Male", [True, False])
embark_town = st.selectbox("Embark Town", ["Southampton", "Cherbourg", "Queenstown"])
alone = st.selectbox("Traveling Alone", [True, False])

if st.button("Predict"):
    input_data = pd.DataFrame([{
        'age': age,
        'fare': fare,
        'sibsp': sibsp,
        'parch': parch,
        'sex': sex,
        'embarked': embarked,
        'class': pclass,
        'who': who,
        'adult_male': adult_male,
        'embark_town': embark_town,
        'alone': alone
    }])

    # Feature Engineering
    input_data['FamilySize'] = input_data['sibsp'] + input_data['parch'] + 1
    input_data['IsAlone'] = (input_data['FamilySize'] == 1).astype(int)

    # Prediction
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    if prediction == 1:
        st.success(f"🎉 You can survive (Probability: {probability:.2f})")
    else:
        st.error(f"❌ Sad to say , you won't survive (Probability: {probability:.2f})")
