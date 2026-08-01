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

# Inputs
age = st.number_input("Age", min_value=18, max_value=65, value=25)

gender = st.selectbox(
    "Gender",
    ["Male", "Female"]
)

experience = st.number_input(
    "Experience (Years)",
    min_value=0,
    max_value=40,
    value=2
)

salary = st.number_input(
    "Salary",
    min_value=10000,
    max_value=100000,
    value=30000
)

department = st.selectbox(
    "Department",
    ["Sales", "IT", "HR", "Finance"]
)

performance = st.slider(
    "Performance Score",
    40,
    100,
    80
)

training = st.number_input(
    "Training Hours",
    min_value=0,
    max_value=200,
    value=20
)

if st.button("Predict"):

    # Encode Gender
    gender = 1 if gender == "Male" else 0

    # Encode Department
    dept_map = {
        "Sales": 0,
        "IT": 1,
        "HR": 2,
        "Finance": 3
    }
    department = dept_map[department]

    # Experience Level
    if experience <= 2:
        exp_level = 0
    elif experience <= 5:
        exp_level = 1
    else:
        exp_level = 2

    # Feature Engineering
    salary_per_experience = salary / max(experience, 1)

    high_performer = 1 if performance >= 80 else 0

    # Create input dataframe
    data = pd.DataFrame([{
        "Age": age,
        "Gender": gender,
        "Experience": experience,
        "Salary": salary,
        "Department": department,
        "PerformanceScore": performance,
        "TrainingHours": training,
        "Salary_Per_Experience": salary_per_experience,
        "High_Performer": high_performer,
        "Experience_Level_Encoded": exp_level
    }])

    # Scale
    data = scaler.transform(data)

    # Predict
    prediction = model.predict(data)[0]

    if prediction == 1:
        st.success("🎉 Employee is likely to be Promoted.")
    else:
        st.error("❌ Employee is NOT likely to be Promoted.")