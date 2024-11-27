import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go
import plotly.express as px


# Set page configuration
st.set_page_config(page_title="Diabetes Risk Analyzer", page_icon="🩺", layout="wide")

# Load machine learning model
@st.cache_resource
def load_model():
    return joblib.load('diabetes_prediction_model.pkl')

model = load_model()

# Prediction function
def predict_diabetes(gender, age, hypertension, heart_disease, smoking_history, bmi, hba1c, blood_glucose):
    # Encode categorical variables
    gender = 1 if gender == 'Female' else 0
    smoking_history = 1 if smoking_history == 'Yes' else 0

    # Prepare input data
    input_data = pd.DataFrame({
        'gender': [gender], 'age': [age], 'hypertension': [hypertension],
        'heart_disease': [heart_disease], 'smoking_history': [smoking_history],
        'bmi': [bmi], 'HbA1c_level': [hba1c], 'blood_glucose_level': [blood_glucose]
    })

    # Make prediction
    prediction = model.predict(input_data)
    prediction_proba = model.predict_proba(input_data)[0][1]
    return prediction[0], prediction_proba

# Risk visualization function
def create_comprehensive_visualization(prediction_proba, bmi, hba1c, blood_glucose):
    fig = go.Figure()

    # Diabetes Risk Gauge
    fig.add_trace(go.Indicator(
        mode="gauge+number",
        value=prediction_proba * 100,
        domain={'x': [0, 0.33], 'y': [0, 1]},
        title={'text': "Diabetes Risk"},
        gauge={
            'axis': {'range': [0, 100]},
            'steps': [
                {'range': [0, 20], 'color': "green"},
                {'range': [20, 60], 'color': "yellow"},
                {'range': [60, 100], 'color': "red"}
            ]
        }
    ))

    # BMI Gauge
    fig.add_trace(go.Indicator(
        mode="gauge+number",
        value=bmi,
        domain={'x': [0.33, 0.66], 'y': [0, 1]},
        title={'text': "BMI"},
        gauge={
            'axis': {'range': [10, 50]},
            'steps': [
                {'range': [10, 18.5], 'color': "blue"},
                {'range': [18.5, 25], 'color': "green"},
                {'range': [25, 30], 'color': "yellow"},
                {'range': [30, 50], 'color': "red"}
            ]
        }
    ))

    # Blood Glucose Gauge
    fig.add_trace(go.Indicator(
        mode="gauge+number",
        value=blood_glucose,
        domain={'x': [0.66, 1], 'y': [0, 1]},
        title={'text': "Glucose Level"},
        gauge={
            'axis': {'range': [0, 500]},
            'steps': [
                {'range': [0, 100], 'color': "green"},
                {'range': [100, 125], 'color': "yellow"},
                {'range': [125, 500], 'color': "red"}
            ]
        }
    ))

    fig.update_layout(height=400, title_text="Comprehensive Health Risk Assessment")
    return fig

# Sidebar for navigation
def sidebar():
    st.sidebar.title("🩺 Diabetes Risk Navigator")
    st.sidebar.info("Comprehensive health risk assessment tool")
    
    menu = ["Assessment", "Risk Factors", "About"]
    choice = st.sidebar.radio("Navigation", menu)
    return choice

# Main assessment page
def assessment_page():
    st.title("🩺 Diabetes Risk Assessment")

    # Two-column layout
    col1, col2 = st.columns(2)

    with col1:
        st.header("📋 Patient Profile")
        gender = st.selectbox("Gender", ["Male", "Female"])
        age = st.slider("Age", 18, 80, 30)
        hypertension = st.checkbox("Hypertension")
        heart_disease = st.checkbox("Heart Disease")
        smoking_history = st.radio("Smoking History", ["Yes", "No"])

    with col2:
        st.header("📊 Health Metrics")
        bmi = st.number_input("BMI", 15.0, 50.0, 25.0, step=0.5)
        hba1c = st.number_input("HbA1c Level (%)", 4.0, 15.0, 5.7, step=0.1)
        blood_glucose = st.number_input("Blood Glucose (mg/dL)", 70, 500, 100)

        # BMI Category
        bmi_categories = {
            (0, 18.5): "Underweight",
            (18.5, 25): "Normal",
            (25, 30): "Overweight",
            (30, 100): "Obese"
        }
        bmi_category = next(cat for (low, high), cat in bmi_categories.items() if low <= bmi < high)
        st.info(f"BMI Category: {bmi_category}")

    # Prediction button
    if st.button("Analyze Risk"):
        prediction, prediction_proba = predict_diabetes(
            gender, age, int(hypertension), int(heart_disease), 
            smoking_history, bmi, hba1c, blood_glucose
        )

        # Comprehensive visualization
        st.plotly_chart(create_comprehensive_visualization(
            prediction_proba, bmi, hba1c, blood_glucose
        ))

        # Risk interpretation
        if prediction == 1:
            st.error(f"High Diabetes Risk: {prediction_proba:.2%} probability")
        else:
            st.success(f"Low Diabetes Risk: {prediction_proba:.2%} probability")

        # Detailed risk profile
        st.header("🔍 Detailed Risk Profile")
        risk_details = {
            "Overall Risk": "High" if prediction == 1 else "Low",
            "Risk Probability": f"{prediction_proba:.2%}",
            "Age Group": f"{age} years",
            "BMI Status": f"{bmi} ({bmi_category})",
            "Additional Factors": [
                "Hypertension" if hypertension else "No Hypertension",
                "Heart Disease" if heart_disease else "No Heart Disease",
                f"Smoking History: {smoking_history}"
            ]
        }

        for key, value in risk_details.items():
            if isinstance(value, list):
                st.markdown(f"**{key}:**")
                for item in value:
                    st.markdown(f"- {item}")
            else:
                st.markdown(f"**{key}:** {value}")

# Risk factors page
def risk_factors_page():
    st.title("🚨 Diabetes Risk Factors")
    
    # Simulated risk factor data
    risk_data = {
        'Factor': ['Age', 'BMI', 'Family History', 'Physical Inactivity', 'High Blood Pressure'],
        'Impact Score': [75, 65, 50, 40, 35]
    }
    df = pd.DataFrame(risk_data)

    # Bar chart of risk factors
    fig = px.bar(
        df, x='Factor', y='Impact Score', 
        title='Diabetes Risk Factors Breakdown',
        color='Impact Score', 
        color_continuous_scale='Viridis'
    )
    st.plotly_chart(fig)

    # Additional risk factor details
    risk_explanations = {
        "Age": "Risk increases with age, particularly after 45",
        "BMI": "Higher BMI correlates with increased diabetes risk",
        "Family History": "Genetic predisposition plays a significant role",
        "Physical Inactivity": "Lack of exercise contributes to metabolic issues",
        "High Blood Pressure": "Associated with increased diabetes likelihood"
    }

    st.header("📝 Risk Factor Insights")
    for factor, explanation in risk_explanations.items():
        st.markdown(f"**{factor}:** {explanation}")

# About page
def about_page():
    st.title("ℹ️ About Diabetes Risk Assessment")
    st.write("""
    ### Comprehensive Health Screening Tool
    - Early detection of diabetes risk
    - Personalized health insights
    - Machine learning powered analysis
    """)
    st.caption("Disclaimer: Not a substitute for professional medical advice")

# Main app logic
def main():
    choice = sidebar()
    
    if choice == "Assessment":
        assessment_page()
    elif choice == "Risk Factors":
        risk_factors_page()
    else:
        about_page()

if __name__ == "__main__":
    main()