import streamlit as st
import pickle
import pandas as pd

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Employee Promotion Prediction",
    page_icon="👨‍💼",
    layout="wide"
)

# ---------------- SIDEBAR ----------------
st.sidebar.title("👨‍💼 Employee Promotion Predictor")
st.sidebar.markdown("""
This application predicts whether an employee is likely to be promoted based on their details.

### Technologies
- Python
- Streamlit
- Scikit-learn
- Pandas

Developed by **Shanu Kumar**
""")

# ---------------- LOAD MODEL ----------------
with open("promotion_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("promotion_scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# ---------------- TITLE ----------------
st.title("👨‍💼 Employee Promotion Prediction")
st.markdown("Fill in the employee information below and click **Predict**.")

# ---------------- INPUTS ----------------
col1, col2 = st.columns(2)

with col1:
    employee_id = st.number_input("Employee ID", min_value=1, value=1001)
    age = st.number_input("Age", 18, 65, 25)
    gender = st.selectbox("Gender", ["Male", "Female"])
    experience = st.number_input("Experience (Years)", 0, 40, 2)

with col2:
    salary = st.number_input("Salary", 10000, 100000, 30000)
    department = st.selectbox(
        "Department",
        ["Sales", "IT", "HR", "Finance"]
    )
    performance = st.slider("Performance Score", 40, 100, 80)
    training = st.number_input("Training Hours", 0, 200, 20)

# ---------------- PREDICT ----------------
if st.button("🚀 Predict Promotion"):

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

    salary_per_experience = salary / max(experience, 1)

    high_performer = 1 if performance >= 80 else 0

    data = pd.DataFrame([{
        "EmployeeID": employee_id,
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

    st.subheader("📋 Employee Summary")
    st.dataframe(data)

    data_scaled = scaler.transform(data)

    prediction = model.predict(data_scaled)[0]

    # Probability (only if available)
    try:
        probability = model.predict_proba(data_scaled)[0][1]

        st.subheader("📊 Promotion Probability")
        st.progress(int(probability * 100))
        st.write(f"**{probability*100:.2f}%** chance of promotion")

    except:
        probability = None

    st.markdown("---")

    if prediction == 1:
        st.balloons()
        st.success("🎉 Employee is likely to be PROMOTED.")
    else:
        st.error("❌ Employee is NOT likely to be promoted.")

    result = pd.DataFrame({
        "Employee ID": [employee_id],
        "Prediction": [
            "Promoted" if prediction == 1 else "Not Promoted"
        ]
    })

    st.download_button(
        "📥 Download Prediction",
        result.to_csv(index=False),
        file_name="prediction.csv",
        mime="text/csv"
    )

# ---------------- FOOTER ----------------
st.markdown("---")
st.markdown(
    "<center>Developed with ❤️ by <b>Shanu Kumar</b></center>",
    unsafe_allow_html=True
)