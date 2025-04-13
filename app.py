import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Set page config
st.set_page_config(
    page_title="NYC Green Taxi Analysis",
    page_icon="🚖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom styling
st.markdown("""
<style>
    .main {
        background-color: #f8f9fa;
    }
    .st-emotion-cache-1v0mbdj {
        border-radius: 10px;
    }
    .stPlot {
        border-radius: 10px;
    }
    .highlight {
        background-color: #e6f3ff;
        padding: 10px;
        border-radius: 5px;
        margin: 10px 0;
    }
    .metric-card {
        background-color: white;
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        margin-bottom: 15px;
    }
</style>
""", unsafe_allow_html=True)

# Load data function with caching
@st.cache_data
def load_data():
    try:
        # Update this path to point to your actual file location
        nycgreentaxi = pd.read_parquet("green_tripdata_2023-01.parquet")
        
        # Data processing
        nycgreentaxi = nycgreentaxi.drop(["ehail_fee", "fare_amount"], axis=1, errors='ignore')
        nycgreentaxi["trip_duration"] = nycgreentaxi.lpep_dropoff_datetime - nycgreentaxi.lpep_pickup_datetime
        nycgreentaxi["trip_duration"] = nycgreentaxi["trip_duration"].dt.total_seconds()/60
        nycgreentaxi["weekday"] = nycgreentaxi["lpep_dropoff_datetime"].dt.day_name()
        nycgreentaxi["hour"] = nycgreentaxi["lpep_dropoff_datetime"].dt.hour
        
        # Missing values imputation
        num_cols = ['trip_distance', 'extra', 'mta_tax', 'tip_amount', 
                    'tolls_amount', 'improvement_surcharge', 'congestion_surcharge', 
                    'trip_duration', 'passenger_count', 'RatecodeID', 'payment_type', 'trip_type']
        for col in num_cols:
            if col in nycgreentaxi.columns and nycgreentaxi[col].isnull().sum() > 0:
                nycgreentaxi[col] = nycgreentaxi[col].fillna(nycgreentaxi[col].median())

        cat_cols = ['store_and_fwd_flag', 'weekday']
        for col in cat_cols:
            if col in nycgreentaxi.columns and nycgreentaxi[col].isnull().sum() > 0:
                nycgreentaxi[col] = nycgreentaxi[col].fillna(nycgreentaxi[col].mode()[0])
        
        return nycgreentaxi
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

# Load data
with st.spinner('Loading and processing data...'):
    nycgreentaxi = load_data()

if nycgreentaxi is not None:
    # App header
    st.title("🚖 NYC Green Taxi Analysis - August 2022")
    st.markdown("""
    Explore insights from New York City's Green Taxi trip records for August 2022. 
    This interactive dashboard visualizes trip patterns, payment methods, and more.
    """)
    
    # Sidebar filters
    st.sidebar.header("Filters")
    
    # Weekday filter
    weekdays = ['All'] + sorted(nycgreentaxi['weekday'].unique().tolist())
    selected_weekday = st.sidebar.selectbox("Select Weekday", weekdays, index=0)
    
    # Hour filter
    hours = ['All'] + sorted(nycgreentaxi['hour'].unique().tolist())
    selected_hour = st.sidebar.selectbox("Select Hour", hours, index=0)
    
    # Payment type filter
    payment_types = ['All'] + sorted(nycgreentaxi['payment_type'].dropna().unique().tolist())
    selected_payment = st.sidebar.selectbox("Select Payment Type", payment_types, index=0)
    
    # Apply filters
    filtered_data = nycgreentaxi.copy()
    if selected_weekday != 'All':
        filtered_data = filtered_data[filtered_data['weekday'] == selected_weekday]
    if selected_hour != 'All':
        filtered_data = filtered_data[filtered_data['hour'] == selected_hour]
    if selected_payment != 'All':
        filtered_data = filtered_data[filtered_data['payment_type'] == selected_payment]
    
    # Key metrics
    st.subheader("📊 Key Metrics")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown('<div class="metric-card">Total Trips: <span style="font-size:24px;color:#1f77b4;">{:,}</span></div>'.format(
            len(filtered_data)), unsafe_allow_html=True)
    
    with col2:
        avg_duration = filtered_data['trip_duration'].mean()
        st.markdown('<div class="metric-card">Avg. Trip Duration: <span style="font-size:24px;color:#ff7f0e;">{:.1f} min</span></div>'.format(
            avg_duration), unsafe_allow_html=True)
    
    with col3:
        avg_distance = filtered_data['trip_distance'].mean()
        st.markdown('<div class="metric-card">Avg. Distance: <span style="font-size:24px;color:#2ca02c;">{:.1f} miles</span></div>'.format(
            avg_distance), unsafe_allow_html=True)
    
    with col4:
        avg_total = filtered_data['total_amount'].mean()
        st.markdown('<div class="metric-card">Avg. Total Amount: <span style="font-size:24px;color:#d62728;">${:.2f}</span></div>'.format(
            avg_total), unsafe_allow_html=True)
    
    # Visualization section
    st.subheader("📈 Data Visualizations")
    
    # Tab layout for different visualizations
    tab1, tab2, tab3, tab4 = st.tabs([
        "Trip Patterns", 
        "Payment & Trip Types", 
        "Hourly Analysis", 
        "Trip Duration vs. Distance"
    ])
    
    with tab1:
        st.markdown("### Trip Patterns by Weekday")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Trip count by weekday
            fig, ax = plt.subplots(figsize=(10, 5))
            weekday_counts = filtered_data['weekday'].value_counts()
            weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            weekday_counts = weekday_counts.reindex(weekday_order)
            weekday_counts.plot(kind='bar', color='#1f77b4', ax=ax)
            ax.set_title('Number of Trips by Weekday')
            ax.set_xlabel('Weekday')
            ax.set_ylabel('Number of Trips')
            plt.xticks(rotation=45)
            st.pyplot(fig)
        
        with col2:
            # Average total amount by weekday
            fig, ax = plt.subplots(figsize=(10, 5))
            avg_amount = filtered_data.groupby('weekday')['total_amount'].mean()
            avg_amount = avg_amount.reindex(weekday_order)
            avg_amount.plot(kind='bar', color='#2ca02c', ax=ax)
            ax.set_title('Average Total Amount by Weekday')
            ax.set_xlabel('Weekday')
            ax.set_ylabel('Average Total Amount ($)')
            plt.xticks(rotation=45)
            st.pyplot(fig)
    
    with tab2:
        st.markdown("### Payment and Trip Type Distribution")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Payment type pie chart
            fig, ax = plt.subplots(figsize=(8, 8))
            payment_counts = filtered_data['payment_type'].value_counts()
            payment_counts.plot(kind='pie', autopct='%1.1f%%', ax=ax, 
                               colors=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
            ax.set_title('Payment Type Distribution')
            ax.set_ylabel('')
            st.pyplot(fig)
        
        with col2:
            # Trip type pie chart
            if 'trip_type' in filtered_data.columns:
                fig, ax = plt.subplots(figsize=(8, 8))
                trip_type_counts = filtered_data['trip_type'].value_counts()
                trip_type_counts.plot(kind='pie', autopct='%1.1f%%', ax=ax,
                                     colors=['#1f77b4', '#ff7f0e'])
                ax.set_title('Trip Type Distribution')
                ax.set_ylabel('')
                st.pyplot(fig)
            else:
                st.warning("Trip type data not available in this dataset")
    
    with tab3:
        st.markdown("### Hourly Trip Patterns")
        
        # Hourly trip count
        fig, ax = plt.subplots(figsize=(12, 6))
        hourly_counts = filtered_data['hour'].value_counts().sort_index()
        hourly_counts.plot(kind='bar', color='#1f77b4', ax=ax)
        ax.set_title('Number of Trips by Hour of Day')
        ax.set_xlabel('Hour of Day')
        ax.set_ylabel('Number of Trips')
        st.pyplot(fig)
        
        # Hourly average trip duration
        fig, ax = plt.subplots(figsize=(12, 6))
        hourly_duration = filtered_data.groupby('hour')['trip_duration'].mean()
        hourly_duration.plot(kind='line', marker='o', color='#d62728', ax=ax)
        ax.set_title('Average Trip Duration by Hour of Day')
        ax.set_xlabel('Hour of Day')
        ax.set_ylabel('Average Duration (minutes)')
        st.pyplot(fig)
    
    with tab4:
        st.markdown("### Relationship Between Trip Duration and Distance")
        
        # Scatter plot with regression line
        fig, ax = plt.subplots(figsize=(10, 6))
        sample_data = filtered_data.sample(min(1000, len(filtered_data)))
        sns.regplot(x='trip_distance', y='trip_duration', data=sample_data, 
                    scatter_kws={'alpha':0.3, 'color':'#1f77b4'}, 
                    line_kws={'color':'#d62728'}, ax=ax)
        ax.set_title('Trip Duration vs. Distance')
        ax.set_xlabel('Distance (miles)')
        ax.set_ylabel('Duration (minutes)')
        st.pyplot(fig)
        
        # Calculate correlation
        corr = filtered_data['trip_distance'].corr(filtered_data['trip_duration'])
        st.markdown(f"""
        <div class="highlight">
            <h4>Correlation Analysis</h4>
            <p>The correlation between trip distance and duration is: <strong>{corr:.2f}</strong></p>
            <p>This suggests a {'strong' if abs(corr) > 0.7 else 'moderate' if abs(corr) > 0.3 else 'weak'} 
            {'positive' if corr > 0 else 'negative'} relationship between distance and duration.</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Data exploration section
    st.subheader("🔍 Explore the Data")
    
    # Show filtered data
    st.markdown(f"**Showing {len(filtered_data)} trips** (filtered from {len(nycgreentaxi)} total trips)")
    st.dataframe(filtered_data.head(100))
    
    # Download filtered data
    csv = filtered_data.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download filtered data as CSV",
        data=csv,
        file_name="nyc_green_taxi_filtered.csv",
        mime="text/csv"
    )
    
    # About section
    st.subheader("ℹ️ About This Dashboard")
    st.markdown("""
    This dashboard analyzes New York City Green Taxi trip records for August 2022. 
    The data includes information about trip durations, distances, payment methods, 
    and other relevant metrics that help understand taxi usage patterns in NYC.
    
    **Key Features:**
    - Interactive filters to explore different segments of the data
    - Visualizations showing trip patterns by time and payment methods
    - Correlation analysis between trip distance and duration
    - Ability to download filtered data for further analysis
    
    **Data Source:** NYC Green Taxi Trip Records (August 2022)
    """)
    
    # Add some space at the bottom
    st.markdown("<br><br>", unsafe_allow_html=True)
else:
    st.error("Failed to load data. Please check the data file and try again.")
