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
    page_title="NYC Green Taxi Analysis - Jan 2023",
    page_icon="🚖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom styling with improved colors and modern look
st.markdown("""
<style>
    .main {
        background-color: #f8f9fa;
    }
    .st-emotion-cache-1v0mbdj {
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    .stPlot {
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    .highlight {
        background-color: #f0f7ff;
        padding: 15px;
        border-radius: 8px;
        margin: 15px 0;
        border-left: 4px solid #1e88e5;
    }
    .metric-card {
        background-color: white;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        margin-bottom: 20px;
        transition: transform 0.3s;
    }
    .metric-card:hover {
        transform: translateY(-5px);
    }
    .title-text {
        color: #1e88e5;
        font-weight: 700;
    }
    .viz-title {
        color: #424242;
        font-weight: 600;
        margin-bottom: 15px;
    }
    .tabs .stTab {
        border-radius: 8px 8px 0 0 !important;
    }
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600&display=swap');
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
</style>
""", unsafe_allow_html=True)

# Load data function with caching
@st.cache_data
def load_data():
    try:
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
with st.spinner('Loading and processing data... This may take a moment for large datasets'):
    nycgreentaxi = load_data()

if nycgreentaxi is not None:
    # App header with updated title
    st.markdown('<h1 class="title-text">🚖 NYC Green Taxi Analysis - January 2023</h1>', unsafe_allow_html=True)
    st.markdown("""
    <div style="font-size:16px; color:#616161; margin-bottom:30px;">
    Explore insights from New York City's Green Taxi trip records for January 2023. 
    This interactive dashboard visualizes trip patterns, payment methods, and more with modern visualizations.
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar filters with improved styling
    st.sidebar.markdown('<div style="font-size:18px; font-weight:600; color:#1e88e5; margin-bottom:20px;">Filters</div>', unsafe_allow_html=True)
    
    # Weekday filter
    weekdays = ['All'] + sorted(nycgreentaxi['weekday'].unique().tolist())
    selected_weekday = st.sidebar.selectbox("Select Weekday", weekdays, index=0, key='weekday')
    
    # Hour filter
    hours = ['All'] + sorted(nycgreentaxi['hour'].unique().tolist())
    selected_hour = st.sidebar.selectbox("Select Hour", hours, index=0, key='hour')
    
    # Payment type filter
    payment_types = ['All'] + sorted(nycgreentaxi['payment_type'].dropna().unique().tolist())
    selected_payment = st.sidebar.selectbox("Select Payment Type", payment_types, index=0, key='payment')
    
    # Apply filters
    filtered_data = nycgreentaxi.copy()
    if selected_weekday != 'All':
        filtered_data = filtered_data[filtered_data['weekday'] == selected_weekday]
    if selected_hour != 'All':
        filtered_data = filtered_data[filtered_data['hour'] == selected_hour]
    if selected_payment != 'All':
        filtered_data = filtered_data[filtered_data['payment_type'] == selected_payment]
    
    # Key metrics with improved cards
    st.markdown('<div class="viz-title">📊 Key Performance Indicators</div>', unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""<div class="metric-card"><div style="font-size:14px; color:#616161;">Total Trips</div><div style="font-size:28px; font-weight:700; color:#1e88e5;">{:,}</div></div>""".format(len(filtered_data)), unsafe_allow_html=True)
    
    with col2:
        avg_duration = filtered_data['trip_duration'].mean()
        st.markdown("""<div class="metric-card"><div style="font-size:14px; color:#616161;">Avg. Trip Duration</div><div style="font-size:28px; font-weight:700; color:#ff6d00;">{:.1f} min</div></div>""".format(avg_duration), unsafe_allow_html=True)
    
    with col3:
        avg_distance = filtered_data['trip_distance'].mean()
        st.markdown("""<div class="metric-card"><div style="font-size:14px; color:#616161;">Avg. Distance</div><div style="font-size:28px; font-weight:700; color:#43a047;">{:.1f} miles</div></div>""".format(avg_distance), unsafe_allow_html=True)
    
    with col4:
        avg_total = filtered_data['total_amount'].mean()
        st.markdown("""<div class="metric-card"><div style="font-size:14px; color:#616161;">Avg. Total Amount</div><div style="font-size:28px; font-weight:700; color:#d81b60;">${:.2f}</div></div>""".format(avg_total), unsafe_allow_html=True)
    
    # Visualization section with modern styling
    st.markdown('<div class="viz-title">📈 Interactive Data Visualizations</div>', unsafe_allow_html=True)
    
    # Tab layout for different visualizations
    tab1, tab2, tab3, tab4 = st.tabs([ "📅 Trip Patterns", "💳 Payment Analysis", "⏱️ Hourly Trends", "📏 Distance vs Duration" ])
    
    with tab1:
        st.markdown('<div class="viz-title">Trip Patterns by Weekday</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig, ax = plt.subplots(figsize=(10, 5))
            weekday_counts = filtered_data['weekday'].value_counts().sort_index()
            ax.bar(weekday_counts.index, weekday_counts.values, color=sns.color_palette("Blues_d", len(weekday_counts)))
            ax.set_title("Trips by Weekday", fontsize=14, weight='bold')
            ax.set_xlabel('Weekday')
            ax.set_ylabel('Number of Trips')
            st.pyplot(fig)
        
        with col2:
            fig, ax = plt.subplots(figsize=(10, 5))
            weekday_duration = filtered_data.groupby('weekday')['trip_duration'].mean().sort_index()
            ax.plot(weekday_duration.index, weekday_duration.values, marker='o', linestyle='-', color='#1e88e5', label='Avg Trip Duration')
            ax.set_title("Avg Trip Duration by Weekday", fontsize=14, weight='bold')
            ax.set_xlabel('Weekday')
            ax.set_ylabel('Avg Duration (min)')
            st.pyplot(fig)
    
    with tab2:
        st.markdown('<div class="viz-title">Payment Type Analysis</div>', unsafe_allow_html=True)
        
        fig, ax = plt.subplots(figsize=(10, 5))
        payment_counts = filtered_data['payment_type'].value_counts()
        ax.pie(payment_counts.values, labels=payment_counts.index, autopct='%1.1f%%', startangle=90, colors=sns.color_palette("Set3", len(payment_counts)))
        ax.set_title("Distribution of Payment Types", fontsize=14, weight='bold')
        st.pyplot(fig)
    
    with tab3:
        st.markdown('<div class="viz-title">Hourly Average Trip Duration</div>', unsafe_allow_html=True)
        
        fig, ax = plt.subplots(figsize=(10, 5))
        hourly_duration = filtered_data.groupby('hour')['trip_duration'].mean()
        heatmap_data = hourly_duration.values.reshape(-1, 1)
        
        cax = ax.imshow(heatmap_data, cmap="coolwarm", aspect="auto", interpolation="nearest")
        ax.set_xticks(np.arange(1))
        ax.set_xticklabels(["Avg Trip Duration"])
        ax.set_yticks(np.arange(len(hourly_duration)))
        ax.set_yticklabels(hourly_duration.index)
        
        fig.colorbar(cax, ax=ax, label='Avg Duration (min)')
        ax.set_title("Hourly Average Trip Duration", fontsize=14, weight='bold')
        st.pyplot(fig)
    
    with tab4:
        st.markdown('<div class="viz-title">Distance vs Duration</div>', unsafe_allow_html=True)
        
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.scatter(filtered_data['trip_distance'], filtered_data['trip_duration'], alpha=0.6, color='#ff6d00')
        ax.set_title("Distance vs Duration", fontsize=14, weight='bold')
        ax.set_xlabel('Trip Distance (miles)')
        ax.set_ylabel('Trip Duration (min)')
        st.pyplot(fig)

# Error handling for failed data loading
else:
    st.error("Failed to load the dataset. Please check the file path and format.")
