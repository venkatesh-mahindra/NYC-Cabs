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
        st.markdown("""
        <div class="metric-card">
            <div style="font-size:14px; color:#616161;">Total Trips</div>
            <div style="font-size:28px; font-weight:700; color:#1e88e5;">{:,}</div>
        </div>""".format(len(filtered_data)), unsafe_allow_html=True)
    
    with col2:
        avg_duration = filtered_data['trip_duration'].mean()
        st.markdown("""
        <div class="metric-card">
            <div style="font-size:14px; color:#616161;">Avg. Trip Duration</div>
            <div style="font-size:28px; font-weight:700; color:#ff6d00;">{:.1f} min</div>
        </div>""".format(avg_duration), unsafe_allow_html=True)
    
    with col3:
        avg_distance = filtered_data['trip_distance'].mean()
        st.markdown("""
        <div class="metric-card">
            <div style="font-size:14px; color:#616161;">Avg. Distance</div>
            <div style="font-size:28px; font-weight:700; color:#43a047;">{:.1f} miles</div>
        </div>""".format(avg_distance), unsafe_allow_html=True)
    
    with col4:
        avg_total = filtered_data['total_amount'].mean()
        st.markdown("""
        <div class="metric-card">
            <div style="font-size:14px; color:#616161;">Avg. Total Amount</div>
            <div style="font-size:28px; font-weight:700; color:#d81b60;">${:.2f}</div>
        </div>""".format(avg_total), unsafe_allow_html=True)
    
    # Visualization section with modern styling
    st.markdown('<div class="viz-title">📈 Interactive Data Visualizations</div>', unsafe_allow_html=True)
    
    # Tab layout for different visualizations
    tab1, tab2, tab3, tab4 = st.tabs([
        "📅 Trip Patterns", 
        "💳 Payment Analysis", 
        "⏱️ Hourly Trends", 
        "📏 Distance vs Duration"
    ])
    
    with tab1:
        st.markdown('<div class="viz-title">Trip Patterns by Weekday</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Trip count by weekday - modern bar chart
            fig, ax = plt.subplots(figsize=(10, 5))
            weekday_counts = filtered_data['weekday'].value_counts()
            weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            weekday_counts = weekday_counts.reindex(weekday_order)
            
            # Modern bar chart with improved styling
            bars = ax.bar(weekday_counts.index, weekday_counts.values, 
                         color=['#1e88e5', '#2196f3', '#64b5f6', '#90caf9', '#bbdefb', '#e3f2fd', '#f5f5f5'],
                         edgecolor='white', linewidth=0.7)
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:,.0f}',
                        ha='center', va='bottom', fontsize=10)
            
            ax.set_title('Number of Trips by Weekday', fontsize=14, pad=20)
            ax.set_xlabel('Weekday', fontsize=12)
            ax.set_ylabel('Number of Trips', fontsize=12)
            ax.grid(axis='y', linestyle='--', alpha=0.7)
            plt.xticks(rotation=45)
            sns.despine()
            st.pyplot(fig)
        
        with col2:
            # Average total amount by weekday - modern line chart
            fig, ax = plt.subplots(figsize=(10, 5))
            avg_amount = filtered_data.groupby('weekday')['total_amount'].mean()
            avg_amount = avg_amount.reindex(weekday_order)
            
            # Modern line chart with markers
            ax.plot(avg_amount.index, avg_amount.values, 
                    marker='o', markersize=8, 
                    color='#ff6d00', linewidth=2.5, 
                    markerfacecolor='white', markeredgewidth=2)
            
            # Add value labels
            for x, y in zip(avg_amount.index, avg_amount.values):
                ax.text(x, y, f'${y:.2f}', 
                        ha='center', va='bottom', fontsize=10)
            
            ax.set_title('Average Total Amount by Weekday', fontsize=14, pad=20)
            ax.set_xlabel('Weekday', fontsize=12)
            ax.set_ylabel('Average Total Amount ($)', fontsize=12)
            ax.grid(axis='y', linestyle='--', alpha=0.7)
            plt.xticks(rotation=45)
            sns.despine()
            st.pyplot(fig)
    
    with tab2:
        st.markdown('<div class="viz-title">Payment and Trip Type Analysis</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Payment type donut chart (modern alternative to pie)
            fig, ax = plt.subplots(figsize=(8, 8))
            payment_counts = filtered_data['payment_type'].value_counts()
            
            # Modern donut chart
            colors = ['#1e88e5', '#ff6d00', '#43a047', '#d81b60']
            wedges, texts, autotexts = ax.pie(payment_counts, 
                                             labels=payment_counts.index,
                                             autopct='%1.1f%%',
                                             startangle=90,
                                             colors=colors,
                                             wedgeprops=dict(width=0.4, edgecolor='w'),
                                             textprops={'fontsize': 12})
            
            # Style the percentages
            for autotext in autotexts:
                autotext.set_color('white')
                autotext.set_fontweight('bold')
            
            # Add a circle at the center to make it a donut
            centre_circle = plt.Circle((0,0),0.60,fc='white')
            ax.add_artist(centre_circle)
            
            ax.set_title('Payment Type Distribution', fontsize=14, pad=20)
            st.pyplot(fig)
        
        with col2:
            # Trip type donut chart (if available)
            if 'trip_type' in filtered_data.columns:
                fig, ax = plt.subplots(figsize=(8, 8))
                trip_type_counts = filtered_data['trip_type'].value_counts()
                
                # Modern donut chart
                colors = ['#1e88e5', '#ff6d00']
                wedges, texts, autotexts = ax.pie(trip_type_counts, 
                                                 labels=trip_type_counts.index,
                                                 autopct='%1.1f%%',
                                                 startangle=90,
                                                 colors=colors,
                                                 wedgeprops=dict(width=0.4, edgecolor='w'),
                                                 textprops={'fontsize': 12})
                
                # Style the percentages
                for autotext in autotexts:
                    autotext.set_color('white')
                    autotext.set_fontweight('bold')
                
                # Add a circle at the center
                centre_circle = plt.Circle((0,0),0.60,fc='white')
                ax.add_artist(centre_circle)
                
                ax.set_title('Trip Type Distribution', fontsize=14, pad=20)
                st.pyplot(fig)
            else:
                st.warning("Trip type data not available in this dataset")
    
    with tab3:
        st.markdown('<div class="viz-title">Hourly Trip Patterns</div>', unsafe_allow_html=True)
        
        # Hourly trip count - area chart
        fig, ax = plt.subplots(figsize=(12, 6))
        hourly_counts = filtered_data['hour'].value_counts().sort_index()
        
        # Modern area chart
        ax.fill_between(hourly_counts.index, hourly_counts.values, 
                        color='#1e88e5', alpha=0.4)
        ax.plot(hourly_counts.index, hourly_counts.values, 
                color='#1e88e5', marker='o', linewidth=2)
        
        ax.set_title('Number of Trips by Hour of Day', fontsize=14, pad=20)
        ax.set_xlabel('Hour of Day', fontsize=12)
        ax.set_ylabel('Number of Trips', fontsize=12)
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        ax.set_xticks(range(0, 24))
        sns.despine()
        st.pyplot(fig)
        
        # Hourly average trip duration - modern line chart
        fig, ax = plt.subplots(figsize=(12, 6))
        hourly_duration = filtered_data.groupby('hour')['trip_duration'].mean()
        
        # Gradient line chart
        points = np.array([hourly_duration.index, hourly_duration.values]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        
        from matplotlib.collections import LineCollection
        lc = LineCollection(segments, cmap='viridis', linewidth=3)
        lc.set_array(hourly_duration.values)
        ax.add_collection(lc)
        
        # Add colorbar
        cbar = plt.colorbar(lc, ax=ax)
        cbar.set_label('Duration (minutes)', rotation=270, labelpad=15)
        
        ax.set_title('Average Trip Duration by Hour of Day', fontsize=14, pad=20)
        ax.set_xlabel('Hour of Day', fontsize=12)
        ax.set_ylabel('Average Duration (minutes)', fontsize=12)
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        ax.set_xticks(range(0, 24))
        ax.set_xlim(0, 23)
        sns.despine()
        st.pyplot(fig)
    
    with tab4:
        st.markdown('<div class="viz-title">Relationship Between Trip Duration and Distance</div>', unsafe_allow_html=True)
        
        # Scatter plot with improved styling
        fig, ax = plt.subplots(figsize=(10, 6))
        sample_data = filtered_data.sample(min(1000, len(filtered_data)))
        
        # Hexbin plot for better visualization of dense areas
        hb = ax.hexbin(x=sample_data['trip_distance'], 
                       y=sample_data['trip_duration'], 
                       gridsize=30, 
                       cmap='viridis', 
                       mincnt=1,
                       edgecolors='none')
        
        # Add colorbar
        cb = fig.colorbar(hb, ax=ax)
        cb.set_label('Number of trips', rotation=270, labelpad=15)
        
        ax.set_title('Trip Duration vs. Distance (Hexbin Plot)', fontsize=14, pad=20)
        ax.set_xlabel('Distance (miles)', fontsize=12)
        ax.set_ylabel('Duration (minutes)', fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.7)
        sns.despine()
        st.pyplot(fig)
        
        # Calculate correlation with improved display
        corr = filtered_data['trip_distance'].corr(filtered_data['trip_duration'])
        st.markdown(f"""
        <div class="highlight">
            <h4 style="color:#1e88e5; margin-top:0;">Correlation Analysis</h4>
            <div style="display:flex; align-items:center; margin-bottom:10px;">
                <div style="font-size:24px; font-weight:700; color:#1e88e5; margin-right:15px;">{corr:.2f}</div>
                <div style="font-size:14px; color:#616161;">
                    The correlation between trip distance and duration is <strong>{'strong' if abs(corr) > 0.7 else 'moderate' if abs(corr) > 0.3 else 'weak'}</strong>
                    and <strong>{'positive' if corr > 0 else 'negative'}</strong>.
                </div>
            </div>
            <div style="background:linear-gradient(90deg, #d81b60 0%, #f8bbd0 50%, #1e88e5 100%); height:8px; border-radius:4px; margin-top:10px;">
                <div style="width:{50 + corr*50}%; height:100%; background-color:transparent;"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Data exploration section with improved layout
    st.markdown('<div class="viz-title">🔍 Explore the Raw Data</div>', unsafe_allow_html=True)
    
    # Show filtered data with expandable section
    with st.expander("View Filtered Data", expanded=False):
        st.markdown(f"**Showing {len(filtered_data):,} trips** (filtered from {len(nycgreentaxi):,} total trips)")
        st.dataframe(filtered_data.head(100).style.set_properties(**{
            'background-color': '#f8f9fa',
            'color': '#424242',
            'border-color': '#e0e0e0'
        }))
    
    # Download filtered data with improved button
    csv = filtered_data.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Filtered Data as CSV",
        data=csv,
        file_name="nyc_green_taxi_jan2023_filtered.csv",
        mime="text/csv",
        key='download-csv',
        help="Download the currently filtered data as a CSV file"
    )
    
    # About section with improved layout
    st.markdown("---")
    st.markdown('<div class="viz-title">ℹ️ About This Dashboard</div>', unsafe_allow_html=True)
    st.markdown("""
    <div style="background-color:#f5f5f5; padding:20px; border-radius:8px;">
        <p style="font-size:15px; line-height:1.6; color:#424242;">
        This dashboard analyzes New York City Green Taxi trip records for <strong>January 2023</strong>. 
        The data includes information about trip durations, distances, payment methods, 
        and other relevant metrics that help understand taxi usage patterns in NYC.
        </p>
        
        <h4 style="color:#1e88e5; margin-top:20px;">Key Features</h4>
        <ul style="font-size:15px; color:#424242;">
            <li>Interactive filters to explore different segments of the data</li>
            <li>Modern, responsive visualizations with improved styling</li>
            <li>Hourly and weekly pattern analysis</li>
            <li>Correlation analysis between trip distance and duration</li>
            <li>Ability to download filtered data for further analysis</li>
        </ul>
        
        <div style="margin-top:20px; font-size:14px; color:#757575;">
            <strong>Data Source:</strong> NYC Green Taxi Trip Records (January 2023)<br>
            <strong>Note:</strong> This is a sample dashboard for demonstration purposes.
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Add some space at the bottom
    st.markdown("<br><br>", unsafe_allow_html=True)
else:
    st.error("Failed to load data. Please check the data file and try again.")
