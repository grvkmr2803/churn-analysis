import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt


st.set_page_config(
    page_title="Customer Churn Prediction (Explainable AI)",
    layout="wide"
)


@st.cache_resource
def load_pipeline():
    return joblib.load("churn_model.pkl")

pipeline = load_pipeline()
preprocessor = pipeline.named_steps["preprocessor"]
model = pipeline.named_steps["model"]


X_train = pd.read_csv("X_train_raw.csv")
X_background = preprocessor.transform(X_train.sample(100, random_state=42))

feature_names = preprocessor.get_feature_names_out()
explainer = shap.TreeExplainer(model, X_background)


st.sidebar.markdown("Customer Profile")

tenure = st.sidebar.slider(
    "Tenure (Months)",
    0, 72, 12,
    help="How long the customer has been with the company"
)

monthly_charges = st.sidebar.slider(
    "Monthly Charges ($)",
    18, 120, 70,
    help="Monthly bill amount"
)

total_charges = st.sidebar.number_input(
    "Total Charges ($)",
    0, 9000, 1000,
    help="Total amount paid by the customer"
)

contract = st.sidebar.selectbox(
    "Contract Type",
    ["Month-to-month", "One year", "Two year"]
)


input_df = pd.DataFrame([{
    "tenure": tenure,
    "MonthlyCharges": monthly_charges,
    "TotalCharges": total_charges,
    "Contract": contract
}])


for col in X_train.columns:
    if col not in input_df.columns:
        input_df[col] = X_train[col].mode()[0]

input_df = input_df[X_train.columns]


prediction_proba = pipeline.predict_proba(input_df)[0][1]


st.markdown(" Customer Churn Prediction (Explainable AI)")
st.write("Predict churn risk and understand **why** using SHAP.")


st.markdown("""
<style>
.kpi-card {
    background: linear-gradient(135deg, #1f4037, #99f2c8);
    padding: 28px;
    border-radius: 18px;
    color: black;
    font-size: 30px;
    font-weight: 700;
    text-align: center;
    margin-bottom: 14px;
}
</style>
""", unsafe_allow_html=True)

st.markdown(
    f"""
    <div class="kpi-card">
        Churn Probability<br>
        {prediction_proba:.2%}
    </div>
    """,
    unsafe_allow_html=True
)


def risk_badge(prob):
    if prob > 0.6:
        return "High Risk"
    elif prob > 0.3:
        return "Medium Risk"
    else:
        return "Low Risk"

st.markdown(f"### Risk Level: **{risk_badge(prediction_proba)}**")


st.divider()
st.markdown("## Why this prediction?")
st.caption("Red increases churn risk · Blue decreases churn risk")

input_transformed = preprocessor.transform(input_df)
input_transformed_df = pd.DataFrame(
    input_transformed,
    columns=feature_names
)

shap_values = explainer(input_transformed_df)

fig, ax = plt.subplots(figsize=(10, 4))
shap.plots.waterfall(
    shap_values[0],
    max_display=8,
    show=False
)
st.pyplot(fig)
plt.close(fig)


st.markdown("###  AI Insight")

top_features = sorted(
    zip(shap_values[0].feature_names, shap_values[0].values),
    key=lambda x: abs(x[1]),
    reverse=True
)[:3]

for name, val in top_features:
    clean_name = name.replace("cat__", "").replace("num__", "").replace("_", " ")
    if val > 0:
        st.write(f"• **{clean_name}** is increasing churn risk")
    else:
        st.write(f"• **{clean_name}** is reducing churn risk")


st.divider()
st.markdown("##  Recommendation")

if prediction_proba > 0.5:
    if tenure < 12:
        st.write("• Offer loyalty discounts or long-term contract incentives")
    if monthly_charges > 80:
        st.write("• Review pricing or bundle services")
else:
    st.write("• Maintain current engagement strategy")

st.markdown("---")
st.caption("Built with XGBoost, SHAP & Streamlit · Explainable AI Demo")
