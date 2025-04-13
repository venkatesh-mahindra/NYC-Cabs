import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import datetime

# Set page configuration
st.set_page_config(
    page_title="NYC Green Taxi Data Analysis",
    page_icon="🚕",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #4CAF50;
        text-align: center;
    }
    .sub-header {
        font-size: 1.8rem;
        color: #2E7D32;
    }
    .section {
        padding: 10px;
        border-radius: 5px;
        margin-bottom: 20px;
    }
    .metric-card {
        background-color: #f5f5f5;
        padding: 15px;
        border-radius: 5px;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("<h1 class='main-header'>NYC Green Taxi Data Analysis</h1>", unsafe_allow_html=True)
st.markdown("### Explore and analyze NYC Green Taxi data from January 2023")

# Function to load and preprocess data
@st.cache_data
def load_data():
    try:
        # Try to load the data
        df = pd.read_parquet('green_tripdata_2023-01.parquet')
        
        # Data preprocessing
        if 'ehail_fee' in df.columns:
            df.drop(columns=['ehail_fee'], inplace=True)
        
        # Calculate trip duration in minutes
        df['trip_duration'] = (df['lpep_dropoff_datetime'] - df['lpep_pickup_datetime']).dt.total_seconds() / 60
        
        # Extract time features
        df['weekday'] = df['lpep_pickup_datetime'].dt.day_name()
        df['day'] = df['lpep_pickup_datetime'].dt.day
        df['hour'] = df['lpep_pickup_datetime'].dt.hour
        df['month'] = df['lpep_pickup_datetime'].dt.month
        df['year'] = df['lpep_pickup_datetime'].dt.year
        df['date'] = df['lpep_pickup_datetime'].dt.date
        
        # Handle missing values
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
        
        object_cols = df.select_dtypes(include=['object']).columns
        df[object_cols] = df[object_cols].fillna('Unknown')
        
        # Filter out extreme values
        df = df[(df['trip_duration'] > 0) & (df['trip_duration'] < 180)]  # Trips between 0 and 3 hours
        df = df[(df['total_amount'] > 0) & (df['total_amount'] < 200)]    # Reasonable fare amounts
        
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

# Load data
with st.spinner("Loading data... This might take a moment."):
    df = load_data()

if df is not None:
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Overview", "Temporal Analysis", "Fare Analysis", "Prediction Model", "Interactive Exploration"])
    
    # Overview page
    if page == "Overview":
        st.markdown("<h2 class='sub-header'>Dataset Overview</h2>", unsafe_allow_html=True)
        
        # Display basic statistics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Trips", f"{len(df):,}")
        with col2:
            st.metric("Average Fare", f"${df['total_amount'].mean():.2f}")
        with col3:
            st.metric("Average Trip Distance", f"{df['trip_distance'].mean():.2f} miles")
        
        # Show sample data
        st.subheader("Sample Data")
        st.dataframe(df.head())
        
        # Data distribution
        st.markdown("<h3 class='sub-header'>Data Distribution</h3>", unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Trip Distance Distribution")
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.histplot(df['trip_distance'], bins=30, kde=True, ax=ax)
            ax.set_title("Trip Distance Distribution")
            ax.set_xlabel("Distance (miles)")
            st.pyplot(fig)
            
        with col2:
            st.subheader("Total Amount Distribution")
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.histplot(df['total_amount'], bins=30, kde=True, ax=ax)
            ax.set_title("Total Amount Distribution")
            ax.set_xlabel("Amount ($)")
            st.pyplot(fig)
        
        # Payment type distribution
        st.subheader("Payment Type Distribution")
        payment_mapping = {1: "Credit Card", 2: "Cash", 3: "No Charge", 4: "Dispute", 5: "Unknown", 6: "Voided Trip"}
        df['payment_type_desc'] = df['payment_type'].map(payment_mapping)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        payment_counts = df['payment_type_desc'].value_counts()
        ax.pie(payment_counts, labels=payment_counts.index, autopct='%1.1f%%', startangle=90, shadow=True)
        ax.set_title("Payment Type Distribution")
        st.pyplot(fig)
        
        # Trip type distribution
        st.subheader("Trip Type Distribution")
        trip_mapping = {1: "Street-hail", 2: "Dispatch"}
        df['trip_type_desc'] = df['trip_type'].map(trip_mapping)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        trip_counts = df['trip_type_desc'].value_counts()
        ax.bar(trip_counts.index, trip_counts.values)
        ax.set_title("Trip Type Distribution")
        ax.set_ylabel("Number of Trips")
        st.pyplot(fig)
    
    # Temporal Analysis page
    elif page == "Temporal Analysis":
        st.markdown("<h2 class='sub-header'>Temporal Analysis</h2>", unsafe_allow_html=True)
        
        # Daily trends
        st.subheader("Daily Trends")
        daily_trips = df.groupby('date').size()
        daily_revenue = df.groupby('date')['total_amount'].sum()
        
        fig, ax1 = plt.subplots(figsize=(12, 6))
        
        color = 'tab:blue'
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Number of Trips', color=color)
        ax1.plot(daily_trips.index, daily_trips.values, color=color)
        ax1.tick_params(axis='y', labelcolor=color)
        
        ax2 = ax1.twinx()
        color = 'tab:red'
        ax2.set_ylabel('Total Revenue ($)', color=color)
        ax2.plot(daily_revenue.index, daily_revenue.values, color=color)
        ax2.tick_params(axis='y', labelcolor=color)
        
        fig.tight_layout()
        plt.title('Daily Trips and Revenue')
        st.pyplot(fig)
        
        # Weekday analysis
        st.subheader("Weekday Analysis")
        
        # Order weekdays properly
        weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        weekday_trips = df['weekday'].value_counts().reindex(weekday_order)
        weekday_avg_fare = df.groupby('weekday')['total_amount'].mean().reindex(weekday_order)
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.bar(weekday_trips.index, weekday_trips.values, color='skyblue')
            ax.set_title('Number of Trips by Weekday')
            ax.set_ylabel('Number of Trips')
            plt.xticks(rotation=45)
            st.pyplot(fig)
            
        with col2:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.bar(weekday_avg_fare.index, weekday_avg_fare.values, color='lightgreen')
            ax.set_title('Average Fare by Weekday')
            ax.set_ylabel('Average Fare ($)')
            plt.xticks(rotation=45)
            st.pyplot(fig)
        
        # Hourly analysis
        st.subheader("Hourly Analysis")
        hourly_trips = df.groupby('hour').size()
        
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(hourly_trips.index, hourly_trips.values, marker='o', linestyle='-', color='purple')
        ax.set_title('Number of Trips by Hour of Day')
        ax.set_xlabel('Hour of Day')
        ax.set_ylabel('Number of Trips')
        ax.set_xticks(range(0, 24))
        ax.grid(True, linestyle='--', alpha=0.7)
        st.pyplot(fig)
        
        # Heatmap: Hour vs Weekday
        st.subheader("Trips by Hour and Weekday")
        
        # Create pivot table
        hour_weekday = pd.crosstab(df['hour'], df['weekday'])
        # Reorder columns to have proper weekday order
        hour_weekday = hour_weekday.reindex(columns=weekday_order)
        
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.heatmap(hour_weekday, cmap="YlGnBu", ax=ax)
        ax.set_title('Number of Trips by Hour and Weekday')
        ax.set_ylabel('Hour of Day')
        ax.set_xlabel('Day of Week')
        st.pyplot(fig)
    
    # Fare Analysis page
    elif page == "Fare Analysis":
        st.markdown("<h2 class='sub-header'>Fare Analysis</h2>", unsafe_allow_html=True)
        
        # Correlation between distance and fare
        st.subheader("Relationship between Trip Distance and Fare")
        
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.scatterplot(x='trip_distance', y='total_amount', data=df.sample(1000), alpha=0.6, ax=ax)
        ax.set_title('Trip Distance vs Total Amount')
        ax.set_xlabel('Trip Distance (miles)')
        ax.set_ylabel('Total Amount ($)')
        st.pyplot(fig)
        
        # Fare components
        st.subheader("Fare Components Breakdown")
        
        # Calculate average values for each component
        fare_components = ['fare_amount', 'tip_amount', 'tolls_amount', 'mta_tax', 
                          'improvement_surcharge', 'congestion_surcharge', 'extra']
        
        avg_components = df[fare_components].mean().sort_values(ascending=False)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(avg_components.index, avg_components.values, color='lightblue')
        ax.set_title('Average Fare Components')
        ax.set_ylabel('Amount ($)')
        plt.xticks(rotation=45)
        st.pyplot(fig)
        
        # Tip analysis
        st.subheader("Tip Analysis")
        
        # Calculate tip percentage
        df['tip_percentage'] = (df['tip_amount'] / df['fare_amount']) * 100
        df['tip_percentage'] = df['tip_percentage'].replace([np.inf, -np.inf], np.nan).fillna(0)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Tip percentage distribution
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.histplot(df[df['tip_percentage'] > 0]['tip_percentage'], bins=20, kde=True, ax=ax)
            ax.set_title('Tip Percentage Distribution (Excluding Zero Tips)')
            ax.set_xlabel('Tip Percentage (%)')
            st.pyplot(fig)
            
        with col2:
            # Average tip by weekday
            avg_tip_weekday = df.groupby('weekday')['tip_percentage'].mean().reindex(weekday_order)
            
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.bar(avg_tip_weekday.index, avg_tip_weekday.values, color='salmon')
            ax.set_title('Average Tip Percentage by Weekday')
            ax.set_ylabel('Tip Percentage (%)')
            plt.xticks(rotation=45)
            st.pyplot(fig)
        
        # Fare by payment type
        st.subheader("Fare Analysis by Payment Type")
        
        avg_fare_payment = df.groupby('payment_type_desc')['total_amount'].mean().sort_values(ascending=False)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(avg_fare_payment.index, avg_fare_payment.values, color='lightgreen')
        ax.set_title('Average Fare by Payment Type')
        ax.set_ylabel('Average Fare ($)')
        plt.xticks(rotation=45)
        st.pyplot(fig)
    
    # Prediction Model page
    elif page == "Prediction Model":
        st.markdown("<h2 class='sub-header'>Fare Prediction Model</h2>", unsafe_allow_html=True)
        
        st.write("""
        This section uses multiple linear regression to predict the total fare amount based on various features.
        The model is trained on the dataset and can be used to estimate fares for new trips.
        """)
        
        # Prepare data for modeling
        st.subheader("Model Training")
        
        # One-hot encode categorical variables
        model_df = df.copy()
        cat_cols = ['weekday', 'payment_type_desc', 'trip_type_desc']
        model_df = pd.get_dummies(model_df, columns=cat_cols, drop_first=True)
        
        # Select features for the model
        features = ['trip_distance', 'trip_duration', 'passenger_count', 'hour'] + \
                  [col for col in model_df.columns if col.startswith(('weekday_', 'payment_type_desc_', 'trip_type_desc_'))]
        
        X = model_df[features]
        y = model_df['total_amount']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train model
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        # Model evaluation
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Mean Squared Error", f"{mse:.4f}")
        with col2:
            st.metric("R² Score", f"{r2:.4f}")
        
        # Feature importance
        st.subheader("Feature Importance")
        
        coef_df = pd.DataFrame({
            'Feature': X.columns,
            'Coefficient': model.coef_
        }).sort_values('Coefficient', ascending=False)
        
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.barplot(x='Coefficient', y='Feature', data=coef_df.head(15), ax=ax)
        ax.set_title('Top 15 Feature Coefficients')
        st.pyplot(fig)
        
        # Prediction vs Actual
        st.subheader("Prediction vs Actual")
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(y_test, y_pred, alpha=0.5)
        ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
        ax.set_xlabel('Actual Fare ($)')
        ax.set_ylabel('Predicted Fare ($)')
        ax.set_title('Actual vs Predicted Fare Amounts')
        st.pyplot(fig)
        
        # Interactive prediction
        st.subheader("Predict Your Fare")
        
        col1, col2 = st.columns(2)
        
        with col1:
            trip_distance = st.slider("Trip Distance (miles)", 0.1, 30.0, 5.0, 0.1)
            trip_duration = st.slider("Trip Duration (minutes)", 1, 120, 15, 1)
            passenger_count = st.slider("Passenger Count", 1, 6, 1, 1)
        
        with col2:
            hour = st.slider("Hour of Day", 0, 23, 12, 1)
            weekday = st.selectbox("Day of Week", weekday_order)
            payment_type = st.selectbox("Payment Type", ["Credit Card", "Cash"])
            trip_type = st.selectbox("Trip Type", ["Street-hail", "Dispatch"])
        
        # Create input data for prediction
        input_data = pd.DataFrame({
            'trip_distance': [trip_distance],
            'trip_duration': [trip_duration],
            'passenger_count': [passenger_count],
            'hour': [hour]
        })
        
        # Add one-hot encoded columns with zeros
        for col in X.columns:
            if col not in input_data.columns:
                input_data[col] = 0
        
        # Set the appropriate one-hot encoded values
        if weekday != 'Monday':  # Assuming 'Monday' was the reference category
            input_data[f'weekday_{weekday}'] = 1
            
        if payment_type == "Credit Card":
            input_data['payment_type_desc_Credit Card'] = 1
            
        if trip_type == "Dispatch":
            input_data['trip_type_desc_Dispatch'] = 1
        
        # Make prediction
        prediction = model.predict(input_data[X.columns])[0]
        
        st.markdown(f"<div class='metric-card'><h3>Estimated Fare: ${prediction:.2f}</h3></div>", unsafe_allow_html=True)
        
        # Daily and weekly predictions
        st.subheader("Daily and Weekly Fare Predictions")
        
        # Daily average prediction
        daily_avg = df.groupby('hour')['total_amount'].mean()
        
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(daily_avg.index, daily_avg.values, marker='o', linestyle='-', color='blue')
        ax.set_title('Average Fare by Hour of Day')
        ax.set_xlabel('Hour of Day')
        ax.set_ylabel('Average Fare ($)')
        ax.set_xticks(range(0, 24))
        ax.grid(True, linestyle='--', alpha=0.7)
        st.pyplot(fig)
        
        # Weekly average prediction
        weekly_avg = df.groupby('weekday')['total_amount'].mean().reindex(weekday_order)
        
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.bar(weekly_avg.index, weekly_avg.values, color='green')
        ax.set_title('Average Fare by Day of Week')
        ax.set_xlabel('Day of Week')
        ax.set_ylabel('Average Fare ($)')
        plt.xticks(rotation=45)
        st.pyplot(fig)
    
    # Interactive Exploration page
    elif page == "Interactive Exploration":
        st.markdown("<h2 class='sub-header'>Interactive Data Exploration</h2>", unsafe_allow_html=True)
        
        st.write("""
        This section allows you to explore relationships between different variables in the dataset.
        Select variables for the x and y axes to create custom visualizations.
        """)
        
        # Select variables for plotting
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
        
        col1, col2 = st.columns(2)
        
        with col1:
            x_var = st.selectbox("X-axis Variable", numeric_cols, index=numeric_cols.index('trip_distance') if 'trip_distance' in numeric_cols else 0)
        
        with col2:
            y_var = st.selectbox("Y-axis Variable", numeric_cols, index=numeric_cols.index('total_amount') if 'total_amount' in numeric_cols else 0)
        
        # Plot options
        plot_type = st.radio("Plot Type", ["Scatter Plot", "Hexbin Plot", "Regression Plot"])
        
        # Create plot
        fig, ax = plt.subplots(figsize=(12, 8))
        
        if plot_type == "Scatter Plot":
            sns.scatterplot(x=x_var, y=y_var, data=df.sample(1000), alpha=0.6, ax=ax)
        elif plot_type == "Hexbin Plot":
            plt.hexbin(df[x_var], df[y_var], gridsize=30, cmap='Blues')
            plt.colorbar(label='Count')
        else:  # Regression Plot
            sns.regplot(x=x_var, y=y_var, data=df.sample(1000), scatter_kws={'alpha':0.5}, line_kws={'color':'red'}, ax=ax)
        
        ax.set_title(f'{y_var} vs {x_var}')
        ax.set_xlabel(x_var)
        ax.set_ylabel(y_var)
        st.pyplot(fig)
        
        # Additional filtering options
        st.subheader("Filter Data")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            min_distance = st.slider("Minimum Trip Distance", 0.0, df['trip_distance'].max(), 0.0, 0.5)
        
        with col2:
            max_fare = st.slider("Maximum Fare Amount", 0.0, 200.0, 100.0, 5.0)
        
        with col3:
            payment_filter = st.multiselect("Payment Types", df['payment_type_desc'].unique().tolist(), df['payment_type_desc'].unique().tolist())
        
        # Apply filters
        filtered_df = df[(df['trip_distance'] >= min_distance) & 
                         (df['total_amount'] <= max_fare) &
                         (df['payment_type_desc'].isin(payment_filter))]
        
        st.write(f"Filtered data contains {len(filtered_df):,} trips")
        
        # Show filtered data statistics
        if not filtered_df.empty:
            st.subheader("Filtered Data Statistics")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Average Distance", f"{filtered_df['trip_distance'].mean():.2f} miles")
            with col2:
                st.metric("Average Fare", f"${filtered_df['total_amount'].mean():.2f}")
            with col3:
                st.metric("Average Duration", f"{filtered_df['trip_duration'].mean():.2f} min")
            
            # Plot filtered data
            fig, ax = plt.subplots(figsize=(12, 8))
            
            if plot_type == "Scatter Plot":
                sns.scatterplot(x=x_var, y=y_var, data=filtered_df.sample(min(1000, len(filtered_df))), alpha=0.6, ax=ax)
            elif plot_type == "Hexbin Plot":
                plt.hexbin(filtered_df[x_var], filtered_df[y_var], gridsize=30, cmap='Blues')
                plt.colorbar(label='Count')
            else:  # Regression Plot
                sns.regplot(x=x_var, y=y_var, data=filtered_df.sample(min(1000, len(filtered_df))), 
                           scatter_kws={'alpha':0.5}, line_kws={'color':'red'}, ax=ax)
            
            ax.set_title(f'{y_var} vs {x_var} (Filtered Data)')
            ax.set_xlabel(x_var)
            ax.set_ylabel(y_var)
            st.pyplot(fig)
else:
    st.error("Failed to load data. Please upload the 'green_tripdata_2023-01.parquet' file.")
    
    # File uploader as a fallback
    uploaded_file = st.file_uploader("Upload the parquet file", type=['parquet'])
    if uploaded_file is not None:
        df = pd.read_parquet(uploaded_file)
        st.success("Data loaded successfully!")
        st.dataframe(df.head())
