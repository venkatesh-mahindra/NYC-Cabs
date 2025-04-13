updated
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

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
    .negative-value {
        color: red;
        font-weight: bold;
    }
    .positive-value {
        color: green;
        font-weight: bold;
    }
    .warning-box {
        background-color: #FFF3CD;
        padding: 10px;
        border-radius: 5px;
        margin: 10px 0;
        border-left: 5px solid #FFC107;
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
        
        # Basic preprocessing
        # Calculate trip duration in minutes
        if 'lpep_pickup_datetime' in df.columns and 'lpep_dropoff_datetime' in df.columns:
            df['trip_duration'] = (df['lpep_dropoff_datetime'] - df['lpep_pickup_datetime']).dt.total_seconds() / 60
        
        # Extract time features
        if 'lpep_pickup_datetime' in df.columns:
            df['weekday'] = df['lpep_pickup_datetime'].dt.day_name()
            df['hour'] = df['lpep_pickup_datetime'].dt.hour
            df['date'] = df['lpep_pickup_datetime'].dt.date
        
        # Handle missing values in numeric columns
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
        if len(numeric_cols) > 0:
            df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())  # Using median instead of mean
        
        # Filter out extreme values
        if 'trip_duration' in df.columns:
            df = df[(df['trip_duration'] > 0) & (df['trip_duration'] < 180)]  # Trips between 0 and 3 hours
        if 'total_amount' in df.columns:
            df = df[(df['total_amount'] > 0) & (df['total_amount'] < 200)]    # Reasonable fare amounts
        if 'trip_distance' in df.columns:
            df = df[(df['trip_distance'] > 0) & (df['trip_distance'] < 50)]    # Reasonable distances
        
        # Remove any remaining negative values in fare and distance
        if 'total_amount' in df.columns:
            df = df[df['total_amount'] > 0]
        if 'trip_distance' in df.columns:
            df = df[df['trip_distance'] > 0]
        
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

# Load data
with st.spinner("Loading data... This might take a moment."):
    df = load_data()

if df is not None:
    # Define weekday order
    weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    
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
            avg_fare = df['total_amount'].mean() if 'total_amount' in df.columns else 0
            st.metric("Average Fare", f"${avg_fare:.2f}")
        with col3:
            avg_distance = df['trip_distance'].mean() if 'trip_distance' in df.columns else 0
            st.metric("Average Trip Distance", f"{avg_distance:.2f} miles")
        
        # Show sample data
        st.subheader("Sample Data")
        st.dataframe(df.head())
        
        # Data distribution
        st.markdown("<h3 class='sub-header'>Data Distribution</h3>", unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            if 'trip_distance' in df.columns:
                st.subheader("Trip Distance Distribution")
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.histplot(df['trip_distance'], bins=30, kde=True, ax=ax)
                ax.set_title("Trip Distance Distribution")
                ax.set_xlabel("Distance (miles)")
                st.pyplot(fig)
            else:
                st.warning("Trip distance data not available")
            
        with col2:
            if 'total_amount' in df.columns:
                st.subheader("Total Amount Distribution")
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.histplot(df['total_amount'], bins=30, kde=True, ax=ax)
                ax.set_title("Total Amount Distribution")
                ax.set_xlabel("Amount ($)")
                st.pyplot(fig)
            else:
                st.warning("Total amount data not available")
        
        # Payment type distribution if available
        if 'payment_type' in df.columns:
            st.subheader("Payment Type Distribution")
            
            # Simple mapping without relying on a new column
            payment_counts = df['payment_type'].value_counts()
            payment_labels = {
                1: "Credit Card", 
                2: "Cash", 
                3: "No Charge", 
                4: "Dispute", 
                5: "Unknown", 
                6: "Voided Trip"
            }
            
            # Replace index with labels if they exist in the mapping
            payment_counts.index = [payment_labels.get(i, str(i)) for i in payment_counts.index]
            
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.pie(payment_counts, labels=payment_counts.index, autopct='%1.1f%%', startangle=90, shadow=True)
            ax.set_title("Payment Type Distribution")
            st.pyplot(fig)
        
        # Trip type distribution if available
        if 'trip_type' in df.columns:
            st.subheader("Trip Type Distribution")
            
            # Simple mapping without relying on a new column
            trip_counts = df['trip_type'].value_counts()
            trip_labels = {1: "Street-hail", 2: "Dispatch"}
            
            # Replace index with labels if they exist in the mapping
            trip_counts.index = [trip_labels.get(i, str(i)) for i in trip_counts.index]
            
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.bar(trip_counts.index, trip_counts.values)
            ax.set_title("Trip Type Distribution")
            ax.set_ylabel("Number of Trips")
            st.pyplot(fig)
    
    # Temporal Analysis page
    elif page == "Temporal Analysis":
        st.markdown("<h2 class='sub-header'>Temporal Analysis</h2>", unsafe_allow_html=True)
        
        if 'date' not in df.columns or 'total_amount' not in df.columns:
            st.warning("Required columns for temporal analysis not found in dataset")
        else:
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
            
            if 'weekday' not in df.columns:
                st.warning("Weekday data not available")
            else:
                # Get weekday counts and ensure all weekdays are represented
                weekday_counts = df['weekday'].value_counts()
                for day in weekday_order:
                    if day not in weekday_counts.index:
                        weekday_counts[day] = 0
                
                # Reindex to ensure correct order
                weekday_counts = weekday_counts.reindex(weekday_order)
                
                # Get average fare by weekday and ensure all weekdays are represented
                weekday_avg_fare = df.groupby('weekday')['total_amount'].mean()
                for day in weekday_order:
                    if day not in weekday_avg_fare.index:
                        weekday_avg_fare[day] = 0
                
                # Reindex to ensure correct order
                weekday_avg_fare = weekday_avg_fare.reindex(weekday_order)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    fig, ax = plt.subplots(figsize=(10, 6))
                    ax.bar(weekday_counts.index, weekday_counts.values, color='skyblue')
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
            if 'hour' not in df.columns:
                st.warning("Hour data not available")
            else:
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
                if 'weekday' in df.columns:
                    st.subheader("Trips by Hour and Weekday")
                    
                    # Create pivot table
                    hour_weekday = pd.crosstab(df['hour'], df['weekday'])
                    
                    # Ensure all weekdays are in the columns
                    for day in weekday_order:
                        if day not in hour_weekday.columns:
                            hour_weekday[day] = 0
                    
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
        
        if 'trip_distance' not in df.columns or 'total_amount' not in df.columns:
            st.warning("Required columns for fare analysis not found in dataset")
        else:
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
            
            # Check which fare components are available
            possible_components = ['fare_amount', 'tip_amount', 'tolls_amount', 'mta_tax', 
                                'improvement_surcharge', 'congestion_surcharge', 'extra']
            available_components = [col for col in possible_components if col in df.columns]
            
            if available_components:
                avg_components = df[available_components].mean().sort_values(ascending=False)
                
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.bar(avg_components.index, avg_components.values, color='lightblue')
                ax.set_title('Average Fare Components')
                ax.set_ylabel('Amount ($)')
                plt.xticks(rotation=45)
                st.pyplot(fig)
            else:
                st.warning("No fare component columns found in the dataset.")
            
            # Tip analysis
            if 'tip_amount' in df.columns and 'fare_amount' in df.columns:
                st.subheader("Tip Analysis")
                
                # Calculate tip percentage safely
                df['tip_percentage'] = np.where(
                    df['fare_amount'] > 0,
                    (df['tip_amount'] / df['fare_amount']) * 100,
                    0
                )
                
                # Replace infinities and NaNs with 0
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
                    if 'weekday' in df.columns:
                        # Calculate average tip percentage by weekday
                        avg_tip_by_weekday = df.groupby('weekday')['tip_percentage'].mean()
                        
                        # Ensure all weekdays are represented
                        for day in weekday_order:
                            if day not in avg_tip_by_weekday.index:
                                avg_tip_by_weekday[day] = 0
                        
                        # Reindex to ensure correct order
                        avg_tip_by_weekday = avg_tip_by_weekday.reindex(weekday_order)
                        
                        fig, ax = plt.subplots(figsize=(10, 6))
                        ax.bar(avg_tip_by_weekday.index, avg_tip_by_weekday.values, color='salmon')
                        ax.set_title('Average Tip Percentage by Weekday')
                        ax.set_ylabel('Tip Percentage (%)')
                        plt.xticks(rotation=45)
                        st.pyplot(fig)
            
            # Fare by payment type
            if 'payment_type' in df.columns:
                st.subheader("Fare Analysis by Payment Type")
                
                # Create a temporary dataframe with payment type labels
                temp_df = df.copy()
                payment_labels = {
                    1: "Credit Card", 
                    2: "Cash", 
                    3: "No Charge", 
                    4: "Dispute", 
                    5: "Unknown", 
                    6: "Voided Trip"
                }
                temp_df['payment_label'] = temp_df['payment_type'].map(payment_labels).fillna('Other')
                
                # Calculate average fare by payment type
                avg_fare_by_payment = temp_df.groupby('payment_label')['total_amount'].mean().sort_values(ascending=False)
                
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.bar(avg_fare_by_payment.index, avg_fare_by_payment.values, color='lightgreen')
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
        
        if 'total_amount' not in df.columns:
            st.warning("Total amount column not found - cannot build prediction model")
        else:
            # Prepare data for modeling
            st.subheader("Model Training")
            
            # Create a copy of the dataframe for modeling
            model_df = df.copy()
            
            # Create dummy variables for weekday if available
            if 'weekday' in model_df.columns:
                weekday_dummies = pd.get_dummies(model_df['weekday'], prefix='weekday', drop_first=True)
                model_df = pd.concat([model_df, weekday_dummies], axis=1)
            
            # Create dummy variables for payment_type if available
            if 'payment_type' in model_df.columns:
                payment_dummies = pd.get_dummies(model_df['payment_type'], prefix='payment', drop_first=True)
                model_df = pd.concat([model_df, payment_dummies], axis=1)
            
            # Create dummy variables for trip_type if available
            if 'trip_type' in model_df.columns:
                trip_dummies = pd.get_dummies(model_df['trip_type'], prefix='trip', drop_first=True)
                model_df = pd.concat([model_df, trip_dummies], axis=1)
            
            # Select features for the model
            base_features = []
            
            # Add available features
            if 'trip_distance' in model_df.columns:
                base_features.append('trip_distance')
            if 'trip_duration' in model_df.columns:
                base_features.append('trip_duration')
            if 'hour' in model_df.columns:
                base_features.append('hour')
            
            # Add passenger_count if available
            if 'passenger_count' in model_df.columns:
                base_features.append('passenger_count')
            
            # Get all dummy columns
            dummy_cols = [col for col in model_df.columns if col.startswith(('weekday_', 'payment_', 'trip_'))]
            
            # Combine all features
            features = base_features + dummy_cols
            
            # Make sure all features exist in the dataframe
            features = [col for col in features if col in model_df.columns]
            
            # Check if we have enough features to build a model
            if len(features) < 2:
                st.warning("Not enough features available to build a prediction model.")
            else:
                X = model_df[features]
                y = model_df['total_amount']
                
                # Scale the features
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
                
                # Split data
                X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
                
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
                    'Feature': features,
                    'Coefficient': model.coef_
                }).sort_values('Coefficient', ascending=False)
                
                # Display feature importance with styling
                def color_coefficient(val):
                    color = 'red' if val < 0 else 'green'
                    return f'color: {color}; font-weight: bold'
                
                st.dataframe(
                    coef_df.style.applymap(color_coefficient, subset=['Coefficient'])\
                    .format({'Coefficient': '{:.4f}'}),
                    height=400
                )
                
                # Plot feature importance
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
                    trip_distance = st.slider("Trip Distance (miles)", 0.1, 30.0, 5.0, 0.1) if 'trip_distance' in features else 5.0
                    trip_duration = st.slider("Trip Duration (minutes)", 1, 120, 15, 1) if 'trip_duration' in features else 15
                    
                    # Only show passenger count slider if it's a feature
                    if 'passenger_count' in features:
                        passenger_count = st.slider("Passenger Count", 1, 6, 1, 1)
                
                with col2:
                    hour = st.slider("Hour of Day", 0, 23, 12, 1) if 'hour' in features else 12
                    weekday = st.selectbox("Day of Week", weekday_order) if any(col.startswith('weekday_') for col in dummy_cols) else None
                    
                    # Only show payment type if it's a feature
                    if any(col.startswith('payment_') for col in dummy_cols):
                        payment_type = st.selectbox("Payment Type", ["Credit Card", "Cash"])
                    
                    # Only show trip type if it's a feature
                    if any(col.startswith('trip_') for col in dummy_cols):
                        trip_type = st.selectbox("Trip Type", ["Street-hail", "Dispatch"])
                
                # Create input data for prediction
                input_data = {}
                
                # Add base features
                if 'trip_distance' in features:
                    input_data['trip_distance'] = trip_distance
                if 'trip_duration' in features:
                    input_data['trip_duration'] = trip_duration
                if 'hour' in features:
                    input_data['hour'] = hour
                if 'passenger_count' in features and 'passenger_count' in locals():
                    input_data['passenger_count'] = passenger_count
                
                # Convert to dataframe
                input_df = pd.DataFrame(input_data, index=[0])
                
                # Add dummy variables with zeros
                for col in dummy_cols:
                    input_df[col] = 0
                
                # Set weekday dummy if available
                if weekday is not None:
                    weekday_col = f'weekday_{weekday}'
                    if weekday_col in dummy_cols:
                        input_df[weekday_col] = 1
                
                # Set payment type dummy if available
                if any(col.startswith('payment_') for col in dummy_cols) and 'payment_type' in locals():
                    payment_mapping = {"Credit Card": 1, "Cash": 2}
                    payment_code = payment_mapping.get(payment_type, 1)
                    
                    for i in range(1, 7):  # Assuming payment types 1-6
                        if i != 1:  # Assuming 1 is the reference category
                            payment_col = f'payment_{i}'
                            if payment_col in dummy_cols:
                                input_df[payment_col] = 1 if i == payment_code else 0
                
                # Set trip type dummy if available
                if any(col.startswith('trip_') for col in dummy_cols) and 'trip_type' in locals():
                    trip_mapping = {"Street-hail": 1, "Dispatch": 2}
                    trip_code = trip_mapping.get(trip_type, 1)
                    
                    for i in range(1, 3):  # Assuming trip types 1-2
                        if i != 1:  # Assuming 1 is the reference category
                            trip_col = f'trip_{i}'
                            if trip_col in dummy_cols:
                                input_df[trip_col] = 1 if i == trip_code else 0
                
                # Make sure input data has all the features used by the model
                for col in features:
                    if col not in input_df.columns:
                        input_df[col] = 0
                
                # Scale the input data
                input_scaled = scaler.transform(input_df[features])
                
                # Make prediction
                try:
                    prediction = model.predict(input_scaled)[0]
                    if prediction < 0:
                        st.markdown(f"""
                        <div class='warning-box'>
                            <h3 class='negative-value'>Estimated Fare: ${prediction:.2f}</h3>
                            <p>Warning: Negative fare predicted. This may indicate:</p>
                            <ul>
                                <li>Unusual input values (very short trip with long duration)</li>
                                <li>Limitations in the model's training data</li>
                                <li>Need for additional features or model refinement</li>
                            </ul>
                            <p>Try adjusting the input values or consider this prediction unreliable.</p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"<div class='metric-card positive-value'><h3>Estimated Fare: ${prediction:.2f}</h3></div>", 
                                  unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"Error making prediction: {e}")
                
                # Model diagnostics
                st.subheader("Model Diagnostics")
                
                # Check for multicollinearity
                st.write("**Multicollinearity Check:**")
                if len(features) > 1:
                    corr_matrix = model_df[features].corr().abs()
                    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
                    high_corr = [column for column in upper.columns if any(upper[column] > 0.7)]
                    if high_corr:
                        st.warning(f"Potential multicollinearity detected between: {', '.join(high_corr)}")
                        st.write("This can make coefficient interpretation unreliable.")
                    else:
                        st.success("No significant multicollinearity detected")
                
                # Check for heteroscedasticity
                residuals = y_test - y_pred
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.scatter(y_pred, residuals, alpha=0.5)
                ax.axhline(y=0, color='r', linestyle='--')
                ax.set_xlabel('Predicted Values')
                ax.set_ylabel('Residuals')
                ax.set_title('Residual Plot (Check for Heteroscedasticity)')
                st.pyplot(fig)
                
                st.write("""
                **Interpretation:**
                - A good model should have residuals randomly scattered around zero
                - Patterns in residuals may indicate issues with the model
                - Negative predictions suggest the model may need more features or better data
                """)
    
    # Interactive Exploration page
    elif page == "Interactive Exploration":
        st.markdown("<h2 class='sub-header'>Interactive Data Exploration</h2>", unsafe_allow_html=True)
        
        st.write("""
        This section allows you to explore relationships between different variables in the dataset.
        Select variables for the x and y axes to create custom visualizations.
        """)
        
        # Select variables for plotting
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
        
        if len(numeric_cols) < 2:
            st.warning("Not enough numeric columns for interactive exploration")
        else:
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
            
            col1, col2 = st.columns(2)
            
            with col1:
                min_distance = st.slider("Minimum Trip Distance", 0.0, df['trip_distance'].max() if 'trip_distance' in df.columns else 0, 0.0, 0.5)
            
            with col2:
                max_fare = st.slider("Maximum Fare Amount", 0.0, 200.0, 100.0, 5.0)
            
            # Create a filter based on payment type if available
            if 'payment_type' in df.columns:
                # Create a temporary column with payment labels
                temp_df = df.copy()
                payment_labels = {
                    1: "Credit Card", 
                    2: "Cash", 
                    3: "No Charge", 
                    4: "Dispute", 
                    5: "Unknown", 
                    6: "Voided Trip"
                }
                temp_df['payment_label'] = temp_df['payment_type'].map(payment_labels).fillna('Other')
                
                # Get unique payment labels
                payment_options = temp_df['payment_label'].unique().tolist()
                
                # Create multiselect for payment types
                payment_filter = st.multiselect("Payment Types", payment_options, payment_options)
                
                # Apply filters
                filtered_df = temp_df[
                    (temp_df['trip_distance'] >= min_distance if 'trip_distance' in temp_df.columns else True) & 
                    (temp_df['total_amount'] <= max_fare if 'total_amount' in temp_df.columns else True) &
                    (temp_df['payment_label'].isin(payment_filter))
                ]
            else:
                # Apply filters without payment type
                filtered_df = df[
                    (df['trip_distance'] >= min_distance if 'trip_distance' in df.columns else True) & 
                    (df['total_amount'] <= max_fare if 'total_amount' in df.columns else True)
                ]
            
            st.write(f"Filtered data contains {len(filtered_df):,} trips")
            
            # Show filtered data statistics
            if not filtered_df.empty:
                st.subheader("Filtered Data Statistics")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    avg_dist = filtered_df['trip_distance'].mean() if 'trip_distance' in filtered_df.columns else 0
                    st.metric("Average Distance", f"{avg_dist:.2f} miles")
                with col2:
                    avg_fare = filtered_df['total_amount'].mean() if 'total_amount' in filtered_df.columns else 0
                    st.metric("Average Fare", f"${avg_fare:.2f}")
                with col3:
                    avg_duration = filtered_df['trip_duration'].mean() if 'trip_duration' in filtered_df.columns else 0
                    st.metric("Average Duration", f"{avg_duration:.2f} min")
                
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
        try:
            df = pd.read_parquet(uploaded_file)
            st.success("Data loaded successfully!")
            st.dataframe(df.head())
        except Exception as e:
            st.error(f"Error loading uploaded file: {e}")
