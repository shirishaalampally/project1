import streamlit as st
import joblib
import pandas as pd

# ================= LOAD =================
model = joblib.load("attrition_model.pkl")
feature_columns = joblib.load("feature_columns.pkl")
lb_encoder = joblib.load("label_encoder.pkl")  # for OverTime

# ================= UI =================
st.title("Employee Attrition Prediction")
st.markdown("Enter employee details to predict if they are likely to leave the company.")

st.sidebar.header("Employee Details")

# ================= INPUT =================
def user_input():
    data = {}

    data['Age'] = st.sidebar.number_input("Age", 18, 65, 30)
    data['MonthlyIncome'] = st.sidebar.number_input("Monthly Income", 1000, 20000, 5000)
    data['JobSatisfaction'] = st.sidebar.selectbox("Job Satisfaction", [1, 2, 3, 4])
    data['OverTime'] = st.sidebar.selectbox("Over Time", ["Yes", "No"])
    data['DistanceFromHome'] = st.sidebar.number_input("Distance From Home", 0, 100, 10)

    return pd.DataFrame([data])

# ================= GET INPUT =================
input_df = user_input()

# ================= ENCODE =================
input_df['OverTime'] = lb_encoder.transform(input_df['OverTime'])

# ================= ALIGN FEATURES =================
input_df = input_df.reindex(columns=feature_columns, fill_value=0)

# ================= PREDICTION =================
if st.button("Predict Attrition"):
    prediction = model.predict(input_df)
    probability = model.predict_proba(input_df)[0][1]

    st.subheader("Result")

    if prediction[0] == 1:
        st.error(f"⚠️ Employee is likely to leave (Probability: {probability:.2f})")
    else:
        st.success(f"✅ Employee will stay (Probability: {1 - probability:.2f})")
        st.info(f"Prediction Probability: {1 - probability:.2f}")

    #st.write("User Input:")
    #st.write(input_df)