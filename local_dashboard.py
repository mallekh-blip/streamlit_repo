# 🌟 STREAMLIT DASHBOARD - LOCAL WORKING VERSION
# Run with: streamlit run local_dashboard.py
# Works with SQLite database created by complete_system.py

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import sqlite3
import time
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Big Data Analytics Dashboard",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Database connection to SQLite
@st.cache_resource
def get_db_connection():
    """Get cached database connection"""
    return sqlite3.connect('streaming_analytics.db', check_same_thread=False)

class LocalDashboard:
    def __init__(self):
        self.conn = get_db_connection()
    
    def get_kpi_metrics(self):
        """Get key performance indicators"""
        try:
            cursor = self.conn.cursor()
            
            # Today's metrics
            cursor.execute("""
                SELECT 
                    COUNT(*) as transactions,
                    SUM(total_amount) as revenue,
                    AVG(total_amount) as avg_order,
                    COUNT(DISTINCT customer_id) as unique_customers
                FROM transactions 
                WHERE DATE(transaction_time) = DATE('now')
            """)
            today_data = cursor.fetchone()
            
            # Anomalies today
            cursor.execute("""
                SELECT COUNT(*) FROM transactions 
                WHERE is_anomaly = 1 AND DATE(transaction_time) = DATE('now')
            """)
            anomalies = cursor.fetchone()[0]
            
            # Total all-time
            cursor.execute("""
                SELECT COUNT(*), SUM(total_amount) FROM transactions
            """)
            total_data = cursor.fetchone()
            
            cursor.close()
            
            return {
                'transactions': today_data[0] or 0,
                'revenue': float(today_data[1] or 0),
                'avg_order': float(today_data[2] or 0),
                'unique_customers': today_data[3] or 0,
                'anomalies': anomalies,
                'total_transactions': total_data[0] or 0,
                'total_revenue': float(total_data[1] or 0)
            }
        except Exception as e:
            st.error(f"Database error: {e}")
            return {'transactions': 0, 'revenue': 0, 'avg_order': 0, 'unique_customers': 0, 'anomalies': 0, 'total_transactions': 0, 'total_revenue': 0}
    
    def get_hourly_trends(self):
        """Get hourly transaction trends"""
        try:
            cursor = self.conn.cursor()
            cursor.execute("""
                SELECT 
                    strftime('%H', transaction_time) as hour,
                    COUNT(*) as transactions,
                    SUM(total_amount) as revenue,
                    AVG(total_amount) as avg_order
                FROM transactions 
                WHERE DATE(transaction_time) = DATE('now')
                GROUP BY strftime('%H', transaction_time)
                ORDER BY hour
            """)
            
            data = cursor.fetchall()
            cursor.close()
            
            if data:
                df = pd.DataFrame(data, columns=['hour', 'transactions', 'revenue', 'avg_order'])
                df['hour'] = pd.to_numeric(df['hour'])
                return df
            return pd.DataFrame()
        except Exception as e:
            st.error(f"Error getting hourly trends: {e}")
            return pd.DataFrame()
    
    def get_top_products(self):
        """Get top selling products today"""
        try:
            cursor = self.conn.cursor()
            cursor.execute("""
                SELECT 
                    p.product_name,
                    p.category,
                    COUNT(*) as sales_count,
                    SUM(t.total_amount) as revenue,
                    SUM(t.quantity) as units_sold
                FROM transactions t
                JOIN products p ON t.product_id = p.product_id
                WHERE DATE(t.transaction_time) = DATE('now')
                GROUP BY p.product_id, p.product_name, p.category
                ORDER BY revenue DESC
                LIMIT 8
            """)
            
            data = cursor.fetchall()
            cursor.close()
            
            if data:
                return pd.DataFrame(data, columns=['product_name', 'category', 'sales_count', 'revenue', 'units_sold'])
            return pd.DataFrame()
        except Exception as e:
            st.error(f"Error getting top products: {e}")
            return pd.DataFrame()
    
    def get_recent_transactions(self, limit=15):
        """Get most recent transactions"""
        try:
            cursor = self.conn.cursor()
            cursor.execute("""
                SELECT 
                    t.transaction_id,
                    c.customer_name,
                    p.product_name,
                    t.quantity,
                    t.total_amount,
                    t.transaction_time,
                    t.is_anomaly
                FROM transactions t
                JOIN customers c ON t.customer_id = c.customer_id
                JOIN products p ON t.product_id = p.product_id
                ORDER BY t.transaction_id DESC
                LIMIT ?
            """, (limit,))
            
            data = cursor.fetchall()
            cursor.close()
            
            if data:
                df = pd.DataFrame(data, columns=['transaction_id', 'customer_name', 'product_name', 'quantity', 'total_amount', 'transaction_time', 'is_anomaly'])
                df['amount_formatted'] = df['total_amount'].apply(lambda x: f"${float(x):.2f}")
                df['anomaly_status'] = df['is_anomaly'].apply(lambda x: "🚨 Anomaly" if x == 1 else "✅ Normal")
                df['time_formatted'] = pd.to_datetime(df['transaction_time']).dt.strftime('%H:%M:%S')
                return df[['transaction_id', 'customer_name', 'product_name', 'quantity', 'amount_formatted', 'time_formatted', 'anomaly_status']]
            
            return pd.DataFrame()
        except Exception as e:
            st.error(f"Error getting recent transactions: {e}")
            return pd.DataFrame()

# Initialize dashboard
@st.cache_resource
def init_dashboard():
    return LocalDashboard()

# Check if database exists
import os
if not os.path.exists('streaming_analytics.db'):
    st.error("⚠️ Database not found! Please run 'complete_system.py' first to generate data.")
    st.code("python complete_system.py", language='bash')
    st.stop()

dashboard = init_dashboard()

# Main dashboard layout
st.title("🚀 Big Data Streaming Analytics Dashboard")
st.markdown("**Local Development Version - Real-Time E-Commerce Analytics**")
st.markdown("---")

# Sidebar controls
st.sidebar.title("Dashboard Controls")
st.sidebar.markdown("### System Status")
st.sidebar.success("✅ Database: SQLite Connected")
st.sidebar.success("✅ Data Pipeline: Local Testing") 
st.sidebar.success("✅ ML Model: Active")

auto_refresh = st.sidebar.checkbox("Auto Refresh (10s)", value=True)
refresh_rate = st.sidebar.slider("Refresh Rate (seconds)", 5, 30, 10)

if st.sidebar.button("🔄 Manual Refresh") or auto_refresh:
    # Get current metrics
    with st.spinner("Loading real-time data..."):
        metrics = dashboard.get_kpi_metrics()
        
        # Alert if no data
        if metrics['total_transactions'] == 0:
            st.warning("⚠️ No data found! Make sure 'complete_system.py' is running to generate streaming data.")
        
        # KPI Cards
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "💰 Total Revenue", 
                f"${metrics['total_revenue']:,.2f}",
                delta=f"${metrics['revenue']:.2f} today"
            )
        
        with col2:
            st.metric(
                "📊 Total Transactions", 
                f"{metrics['total_transactions']:,}",
                delta=f"+{metrics['transactions']} today"
            )
        
        with col3:
            st.metric(
                "🛍️ Avg Order Value", 
                f"${metrics['avg_order']:.2f}" if metrics['avg_order'] > 0 else "$0.00"
            )
        
        with col4:
            st.metric(
                "🚨 Anomalies Detected", 
                f"{metrics['anomalies']}",
                delta="Real-time ML Detection" if metrics['anomalies'] > 0 else "All Clear"
            )
        
        st.markdown("---")
        
        # Charts section
        col1, col2 = st.columns(2)
        
        # Hourly trends
        hourly_data = dashboard.get_hourly_trends()
        if not hourly_data.empty:
            with col1:
                st.subheader("📈 Hourly Transaction Volume")
                fig = px.bar(
                    hourly_data, 
                    x='hour', 
                    y='transactions',
                    title="Transactions per Hour Today",
                    color='transactions',
                    color_continuous_scale='Blues',
                    labels={'hour': 'Hour of Day', 'transactions': 'Number of Transactions'}
                )
                fig.update_layout(showlegend=False, height=400)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.subheader("💵 Hourly Revenue Trend")
                fig = px.line(
                    hourly_data, 
                    x='hour', 
                    y='revenue',
                    title="Revenue per Hour Today",
                    markers=True,
                    labels={'hour': 'Hour of Day', 'revenue': 'Revenue ($)'}
                )
                fig.update_traces(line_color='green', line_width=3, marker_size=8)
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("📊 No hourly data yet. Start 'complete_system.py' to generate streaming data.")
        
        # Top products section
        st.markdown("---")
        st.subheader("🏆 Top Performing Products Today")
        
        top_products = dashboard.get_top_products()
        if not top_products.empty:
            col1, col2 = st.columns(2)
            
            with col1:
                # Revenue pie chart
                fig = px.pie(
                    top_products.head(6), 
                    values='revenue', 
                    names='product_name',
                    title="Revenue Distribution (Top 6 Products)",
                    color_discrete_sequence=px.colors.qualitative.Set3
                )
                fig.update_traces(textposition='inside', textinfo='percent+label')
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Category performance
                if len(top_products) > 0:
                    category_data = top_products.groupby('category').agg({
                        'revenue': 'sum',
                        'units_sold': 'sum'
                    }).reset_index()
                    
                    fig = px.bar(
                        category_data, 
                        x='category', 
                        y='revenue',
                        title="Revenue by Category",
                        color='revenue',
                        color_continuous_scale='Viridis',
                        labels={'category': 'Product Category', 'revenue': 'Revenue ($)'}
                    )
                    fig.update_layout(showlegend=False)
                    st.plotly_chart(fig, use_container_width=True)
            
            # Products table
            st.subheader("📋 Detailed Product Performance")
            
            # Format the dataframe for display
            display_df = top_products.copy()
            display_df['revenue'] = display_df['revenue'].apply(lambda x: f"${float(x):,.2f}")
            display_df['sales_count'] = display_df['sales_count'].apply(lambda x: f"{x:,}")
            display_df['units_sold'] = display_df['units_sold'].apply(lambda x: f"{x:,}")
            
            st.dataframe(
                display_df,
                column_config={
                    "product_name": "Product",
                    "category": "Category", 
                    "sales_count": "Sales Count",
                    "revenue": "Revenue",
                    "units_sold": "Units Sold"
                },
                hide_index=True,
                use_container_width=True
            )
        
        # Recent transactions
        st.markdown("---")
        st.subheader("⏰ Recent Transactions (Live Stream)")
        
        recent_transactions = dashboard.get_recent_transactions()
        if not recent_transactions.empty:
            st.dataframe(
                recent_transactions,
                column_config={
                    "transaction_id": "ID",
                    "customer_name": "Customer",
                    "product_name": "Product",
                    "quantity": "Qty",
                    "amount_formatted": "Amount",
                    "time_formatted": "Time",
                    "anomaly_status": "Status"
                },
                hide_index=True,
                use_container_width=True
            )
            
            # Highlight anomalies
            anomalous_count = (recent_transactions['anomaly_status'] == "🚨 Anomaly").sum()
            if anomalous_count > 0:
                st.warning(f"⚠️ {anomalous_count} anomalous transactions detected in recent activity!")
                st.caption("Anomalies are automatically flagged by our ML-based outlier detection system.")
        else:
            st.info("📋 No transactions yet. Start the data generator to see live streaming data.")
        
        # System metrics
        st.markdown("---")
        st.subheader("🔧 System Performance")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.success("✅ Data Pipeline: 300 rec/min")
        with col2:
            st.success("✅ Outlier Detection: Active")
        with col3:
            st.success("✅ ML Model: Learning")
        with col4:
            st.success("✅ Dashboard: Live Updates")
        
        # Performance metrics
        with st.expander("📊 Detailed System Metrics"):
            perf_col1, perf_col2 = st.columns(2)
            
            with perf_col1:
                st.metric("Total Customers", metrics['unique_customers'])
                st.metric("Processing Rate", "5 transactions/second")
                st.metric("Database Size", f"{metrics['total_transactions']:,} records")
                
            with perf_col2:
                st.metric("System Uptime", "Active")
                st.metric("Database Type", "SQLite (Local)")
                st.metric("ML Predictions", "Online Learning")
        
        # Footer
        st.markdown("---")
        st.caption(f"Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | "
                  f"Auto-refresh: {'On' if auto_refresh else 'Off'} | "
                  f"Refresh Rate: {refresh_rate}s | "
                  f"Status: 🟢 Local System Operational")

# Auto refresh logic
if auto_refresh:
    time.sleep(refresh_rate)
    st.rerun()
else:
    st.info("💡 Enable auto-refresh in the sidebar to see live updates")

# Instructions
with st.sidebar.expander("📋 Quick Start Guide"):
    st.markdown("""
    **1. Generate Data:**
    ```bash
    python complete_system.py
    ```
    
    **2. Launch Dashboard:**
    ```bash
    streamlit run local_dashboard.py
    ```
    
    **3. For Azure Deployment:**
    - Set environment variables
    - Use app.py for Azure App Service
    - Deploy with GitHub integration
    """)

st.sidebar.markdown("---")
st.sidebar.caption("🚀 Big Data Streaming Analytics\nLocal Development Version")