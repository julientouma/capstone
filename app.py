import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import OneHotEncoder
import seaborn as sns
import matplotlib.pyplot as plt
# Set Streamlit page configuration
st.set_page_config(page_title="Financial Aid Predictor", layout="centered")
st.header("📊 Data Visualizations")
df = pd.read_csv("cleanedd.csv")
with st.expander("View Pie Charts for Special Circumstances and Documents"):
    # Pie Charts for specific categorical columns
    pie_columns = ["Special Family Circumstances", "Loan", "FAID Missing Document"]
    fig1, axes1 = plt.subplots(1, 3, figsize=(18, 6))

    for i, col in enumerate(pie_columns):
        data_counts = df[col].astype(str).value_counts()
        axes1[i].pie(data_counts, labels=data_counts.index, autopct='%1.1f%%', colors=plt.cm.Pastel1.colors)
        axes1[i].set_title(f"Distribution of {col}")

    st.pyplot(fig1)

with st.expander("View Count Plots for Applicant Info"):
    # Count Plots for categorical distribution
    cat_plot_columns = ["LEVEL", "Nationality", "Marital Status", "Plan to Reside", "Father Status", "Mother Status"]
    sns.set_style("whitegrid")
    fig2, axes2 = plt.subplots(nrows=3, ncols=2, figsize=(12, 10))
    axes2 = axes2.flatten()

    for i, col in enumerate(cat_plot_columns):
        sns.countplot(x=df[col], data=df, palette="pastel", ax=axes2[i])
        axes2[i].set_title(f"Distribution of {col}")
        axes2[i].set_xlabel("")
        axes2[i].set_ylabel("Count")
        axes2[i].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    st.pyplot(fig2)
# Load model and encoder
model = joblib.load("gradient_boosting_model.pkl")
encoder_path = "onehot_encoder.pkl"
@st.cache_resource
def load_encoder():
    df = pd.read_csv("cleanedd.csv")  # path to your full dataset
    encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
    encoder.fit(df[categorical])
    return encoder

# Define features
categorical = ['LEVEL', 'Nationality', 'Marital Status', 'Plan to Reside', 
               'Father Citizenship', 'Father Status', 'Mother Status', 'Travel Records', 
               'Car_Models', 'FAID_Record_owner']

numerical = ['Father Income Document', 'Mother Income Document', 'Number of Siblings @ AUB',
             'Number of Dependents', 'Number of Properties', 'Total Estimated Value', 'Total Area',
             'Total Tuition for siblings', 'Total Financial Assistance for siblings',
             'Financial Assistant', 'Father_Total_Benefits', 'Mother_Total_Benefits',
             'Number_of_Cars', 'Applicant Annual Income']

encoder = load_encoder()

# Streamlit UI
st.title("🎓 Financial Aid % Prediction")
st.markdown("Enter applicant information below to predict the **percentage of tuition** that will be awarded as financial aid.")

st.header("🧾 Applicant Information")

# Input collection
user_input = {}
for col in categorical:
    options = encoder.categories_[categorical.index(col)]
    user_input[col] = st.selectbox(col, options)

for col in numerical:
    user_input[col] = st.number_input(col, min_value=0.0, step=1.0)

# Predict
if st.button("Predict"):
    input_df = pd.DataFrame([user_input])

    # One-hot encode
    encoded_input = pd.DataFrame(encoder.transform(input_df[categorical]))
    encoded_input.columns = encoder.get_feature_names_out(categorical)

    # Combine with numeric
    final_input = pd.concat([input_df[numerical], encoded_input], axis=1)
    final_input = final_input.apply(pd.to_numeric, errors='coerce').fillna(0)

    # Predict using regression model
    prediction = model.predict(final_input)[0]

    # Display result
    st.subheader("🎯 Prediction Result")
    st.success(f"The estimated financial aid award is: **{prediction:.2f}%** of tuition.")
