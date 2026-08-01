import streamlit as st
import pickle
import pandas as pd

# Load model and scaler
with open("promotion_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("promotion_scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

st.set_page_config(page_title="Employee Promotion Prediction")

st.title("👨‍💼 Employee Promotion Prediction")

age = st.number_input("Age", 18, 65, 25)

gender = st.selectbox("Gender", ["Male", "Female"])

experience = st.number_input("Experience (Years)", 0, 40, 2)

salary = st.number_input("Salary", 10000, 500000, 30000)

department = st.selectbox(
    "Department",
    ["Sales", "IT", "HR", "Finance"]
)

performance = st.slider("Performance Score", 1, 10, 5)

training = st.number_input("Training Hours", 0, 200, 20)

if st.button("Predict"):

    gender = 1 if gender == "Male" else 0

    dept_map = {
        "Sales": 0,
        "IT": 1,
        "HR": 2,
        "Finance": 3
    }

    department = dept_map[department]

    if experience <= 2:
        exp_level = 0
    elif experience <= 5:
        exp_level = 1
    else:
        exp_level = 2

    salary_per_exp = salary / max(experience, 1)

    high_performer = 1 if performance >= 8 else 0

    data = pd.DataFrame([[
        age,
        gender,
        experience,
        salary,
        department,
        performance,
        training,
        exp_level,
        salary_per_exp,
        high_performer
    ]], columns=[
        "Age",
        "Gender",
        "Experience",
        "Salary",
        "Department",
        "PerformanceScore",
        "TrainingHours",
        "Experience_Level_Encoded",
        "Salary_Per_Experience",
        "High_Performer"
    ])

    data = scaler.transform(data)

    prediction = model.predict(data)[0]

    if prediction == 1:
        st.success("🎉 Employee is likely to be Promoted.")
    else:
        st.error("❌ Employee is not likely to be Promoted.")
