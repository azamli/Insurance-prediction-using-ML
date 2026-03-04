import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score
from xgboost import XGBRegressor

# -------------------------
# PAGE CONFIG
# -------------------------

st.set_page_config(page_title="InsureAI", layout="wide")

st.title("🧠 InsureAI – Health Insurance Premium Prediction")
print("Created by Azam Ali")

# -------------------------
# LOAD DATA
# -------------------------

df = pd.read_excel("premiums_rest.xlsx")
df.columns = df.columns.str.replace(" ","_").str.lower()

# -------------------------
# DATA CLEANING
# -------------------------

df.dropna(inplace=True)

# -------------------------
# FEATURE ENGINEERING
# -------------------------

risk_scores = {
"diabetes":6,
"heart disease":8,
"high blood pressure":6,
"thyroid":5,
"no disease":0,
"none":0
}

df[['disease1','disease2']] = df['medical_history'].str.split(" & ", expand=True)
df[['disease1','disease2']] = df[['disease1','disease2']].fillna("none")

df['disease1'] = df['disease1'].str.lower()
df['disease2'] = df['disease2'].str.lower()

df["total_risk_score"] = df["disease1"].map(risk_scores) + df["disease2"].map(risk_scores)

df["normalized_risk_score"] = (
(df["total_risk_score"] - df["total_risk_score"].min()) /
(df["total_risk_score"].max() - df["total_risk_score"].min())
)

# -------------------------
# ENCODING
# -------------------------

df["insurance_plan"] = df["insurance_plan"].map({
"Bronze":1,
"Silver":2,
"Gold":3
})

df["income_level"] = df["income_level"].map({
"<10L":1,
"10L - 25L":2,
"25L - 40L":3,
"> 40L":4
})

df = pd.get_dummies(df,
columns=[
"gender",
"region",
"marital_status",
"bmi_category",
"smoking_status",
"employment_status"
],
drop_first=True)

df = df.drop([
"medical_history",
"disease1",
"disease2",
"total_risk_score"
],axis=1)

# -------------------------
# MODEL TRAINING
# -------------------------

X = df.drop("annual_premium_amount",axis=1)
y = df["annual_premium_amount"]

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

X_train,X_test,y_train,y_test = train_test_split(
X_scaled,y,test_size=0.3,random_state=42
)

model = XGBRegressor(
n_estimators=50,
max_depth=5,
learning_rate=0.1
)

model.fit(X_train,y_train)

r2 = r2_score(y_test,model.predict(X_test))

# -------------------------
# USER INPUT
# -------------------------

st.sidebar.header("User Input")

age = st.sidebar.slider("Age",18,80,30)
income = st.sidebar.slider("Income Lakhs",1,100,10)

gender = st.sidebar.selectbox("Gender",["Male","Female"])

insurance_plan = st.sidebar.selectbox(
"Insurance Plan",
["Bronze","Silver","Gold"]
)

bmi = st.sidebar.number_input("BMI",10.0,50.0,25.0)

smoker = st.sidebar.selectbox("Smoker",["Yes","No"])

obesity = st.sidebar.selectbox("Obesity",["Yes","No"])

risk_score = st.sidebar.slider("Risk Score",0.0,1.0,0.3)

plan_map = {"Bronze":1,"Silver":2,"Gold":3}

input_dict = {col:0 for col in X.columns}

input_dict["age"] = age
input_dict["income_lakhs"] = income
input_dict["insurance_plan"] = plan_map[insurance_plan]
input_dict["normalized_risk_score"] = risk_score

input_df = pd.DataFrame([input_dict])
input_scaled = scaler.transform(input_df)

# -------------------------
# PREDICTION
# -------------------------

if st.sidebar.button("Predict Premium"):

    prediction = model.predict(input_scaled)[0]

    st.subheader("Predicted Annual Premium")

    st.markdown(
    f"""
    <div style="
    background-color:#0d6efd;
    padding:25px;
    border-radius:10px;
    text-align:center;
    color:white;
    font-size:35px">

    ₹ {prediction:,.2f}

    </div>
    """,
    unsafe_allow_html=True
    )

# -------------------------
# MODEL PERFORMANCE
# -------------------------

st.subheader("Model Performance")
st.write("R² Score:",r2)

# -------------------------
# AGE DISTRIBUTION
# -------------------------

st.subheader("Age Distribution")

fig = px.histogram(df,x="age")
st.plotly_chart(fig,use_container_width=True)

# -------------------------
# INCOME VS PREMIUM
# -------------------------

st.subheader("Income vs Premium")

fig = px.scatter(df,x="income_lakhs",y="annual_premium_amount")
st.plotly_chart(fig,use_container_width=True)

# -------------------------
# CORRELATION HEATMAP
# -------------------------

st.subheader("Feature Correlation Heatmap")

fig,ax = plt.subplots(figsize=(12,8))
sns.heatmap(df.corr(),ax=ax)

st.pyplot(fig)

# -------------------------
# RESIDUAL ANALYSIS
# -------------------------

st.subheader("Residual Analysis")

y_pred = model.predict(X_test)
residuals = y_pred - y_test

fig,ax = plt.subplots()
sns.histplot(residuals,kde=True,ax=ax)

st.pyplot(fig)

# -------------------------
# DATASET PREVIEW
# -------------------------

st.subheader("Dataset Preview")
st.dataframe(df.head())
