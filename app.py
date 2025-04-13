import streamlit as st
st.set_page_config(page_title="NYC Taxi Fare Prediction", layout="wide")  # FIRST Streamlit command

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import joblib

# Load preprocessed data
@st.cache_data
def load_data():
    df = pd.read_parquet('green_tripdata_2023-01.parquet')
    df.drop(columns=['ehail_fee'], inplace=True)
    df['trip_duration'] = (df['lpep_dropoff_datetime'] - df['lpep_pickup_datetime']).dt.total_seconds() / 60
    df['weekday'] = df['lpep_dropoff_datetime'].dt.day_name()
    df['hourofday'] = df['lpep_dropoff_datetime'].dt.hour

    # Fill missing values
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
    object_cols = df.select_dtypes(include=['object']).columns
    df[object_cols] = df[object_cols].fillna('Unknown')

    return df

df = load_data()

# Feature encoding
object_cols = ['store_and_fwd_flag', 'RatecodeID', 'payment_type',
               'trip_type', 'weekday', 'hourofday']
df_encoded = pd.get_dummies(df, columns=object_cols)

# Modeling
X = df_encoded.drop(columns=['total_amount', 'lpep_pickup_datetime', 'lpep_dropoff_datetime'])
y = df_encoded['total_amount']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model_lr = LinearRegression()
model_lr.fit(X_train, y_train)

# Prediction function
def predict_fare(hour, passenger_count):
    input_data = np.zeros(X_train.shape[1])
    hour_col = f'hourofday_{hour}'
    if hour_col in X_train.columns:
        idx_hour = X_train.columns.get_loc(hour_col)
        input_data[idx_hour] = 1
    idx_pass = X_train.columns.get_loc('passenger_count')
    input_data[idx_pass] = passenger_count
    return model_lr.predict([input_data])[0]

# UI
st.title("🚖 NYC Green Taxi Fare Prediction App")

tab1, tab2, tab3 = st.tabs(["🔢 Predict Fare", "📈 Daily/Weekly/Monthly", "📊 Visual Analysis"])

# --- Tab 1: Predict Fare ---
with tab1:
    st.subheader("Enter Trip Details to Predict Fare")
    hour = st.slider("Pickup Hour", 0, 23, 10)
    passenger_count = st.number_input("Passenger Count", min_value=1, max_value=6, value=1)

    if st.button("Predict Fare"):
        pred = predict_fare(hour, passenger_count)
        st.success(f"Predicted Total Amount: ${pred:.2f}")

# --- Tab 2: Trend Analysis ---
with tab2:
    st.subheader("Fare Trend Analysis")

    df['date'] = df['lpep_dropoff_datetime'].dt.date
    df['week'] = df['lpep_dropoff_datetime'].dt.isocalendar().week
    df['month'] = df['lpep_dropoff_datetime'].dt.month_name()

    option = st.radio("Select Time Granularity", ["Daily", "Weekly", "Monthly"])

    if option == "Daily":
        daily_avg = df.groupby('date')['total_amount'].mean()
        st.line_chart(daily_avg)
    elif option == "Weekly":
        weekly_avg = df.groupby('week')['total_amount'].mean()
        st.bar_chart(weekly_avg)
    elif option == "Monthly":
        monthly_avg = df.groupby('month')['total_amount'].mean()
        st.bar_chart(monthly_avg)

# --- Tab 3: Visual Analysis ---
with tab3:
    st.subheader("Correlations and Distributions")

    st.markdown("### 🔁 Correlation Heatmap")
    numeric_cols = ['trip_distance', 'fare_amount', 'extra', 'mta_tax', 'tip_amount',
                    'tolls_amount', 'improvement_surcharge', 'congestion_surcharge',
                    'trip_duration', 'passenger_count', 'total_amount']
    corr = df[numeric_cols].corr()

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax)
    st.pyplot(fig)

    st.markdown("### 💰 Distribution of Total Fare")
    fig2, ax2 = plt.subplots(figsize=(8, 4))
    sns.histplot(df['total_amount'], bins=50, kde=True, ax=ax2)
    st.pyplot(fig2)
