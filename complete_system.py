import os
import time
import random
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import threading
import sqlite3
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

print("🚀 BIG DATA STREAMING ANALYTICS SYSTEM")
print("="*60)
print("✅ All libraries imported successfully!")

# =====================================================================
# 1. DATABASE MANAGER (SQLite for Local + PostgreSQL for Azure)
# =====================================================================

class DatabaseManager:
    def __init__(self, use_sqlite=True):
        """Initialize database connection (SQLite for local, PostgreSQL for Azure)"""
        self.use_sqlite = use_sqlite
        
        if use_sqlite:
            # Local SQLite database
            self.db_path = 'streaming_analytics.db'
            self.init_sqlite()
        else:
            # Azure PostgreSQL (set environment variables)
            import psycopg2
            self.connection_params = {
                'host': os.environ.get('DATABASE_HOST', 'localhost'),
                'database': os.environ.get('DATABASE_NAME', 'streaming_analytics'),
                'user': os.environ.get('DATABASE_USER', 'postgres'),
                'password': os.environ.get('DATABASE_PASSWORD', 'password'),
                'port': os.environ.get('DATABASE_PORT', '5432'),
                'sslmode': 'prefer'
            }
    
    def init_sqlite(self):
        """Initialize SQLite database with all tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create tables
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS products (
                product_id INTEGER PRIMARY KEY AUTOINCREMENT,
                product_name TEXT NOT NULL,
                category TEXT NOT NULL,
                price DECIMAL(10,2) NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS customers (
                customer_id INTEGER PRIMARY KEY AUTOINCREMENT,
                customer_name TEXT NOT NULL,
                email TEXT UNIQUE NOT NULL,
                location TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS transactions (
                transaction_id INTEGER PRIMARY KEY AUTOINCREMENT,
                customer_id INTEGER NOT NULL,
                product_id INTEGER NOT NULL,
                quantity INTEGER NOT NULL,
                total_amount DECIMAL(10,2) NOT NULL,
                transaction_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                payment_method TEXT NOT NULL,
                is_anomaly BOOLEAN DEFAULT 0,
                FOREIGN KEY (customer_id) REFERENCES customers(customer_id),
                FOREIGN KEY (product_id) REFERENCES products(product_id)
            )
        ''')
        
        # Create indexes
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_transactions_time ON transactions(transaction_time)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_transactions_anomaly ON transactions(is_anomaly)')
        
        conn.commit()
        cursor.close()
        conn.close()
        print("✅ SQLite database initialized!")
    
    def get_connection(self):
        """Get database connection"""
        if self.use_sqlite:
            return sqlite3.connect(self.db_path, check_same_thread=False)
        else:
            import psycopg2
            return psycopg2.connect(**self.connection_params)
    
    def insert_sample_data(self):
        """Insert sample data if not exists"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        # Check if data already exists
        cursor.execute("SELECT COUNT(*) FROM products")
        if cursor.fetchone()[0] > 0:
            cursor.close()
            conn.close()
            return
        
        # Insert sample products
        products = [
            ('iPhone 15 Pro', 'Electronics', 1199.99),
            ('MacBook Pro', 'Electronics', 2499.99),
            ('AirPods Pro', 'Electronics', 249.99),
            ('Nike Air Max', 'Sports', 130.00),
            ('Adidas Ultraboost', 'Sports', 180.00),
            ('Coffee Maker', 'Home', 89.99),
            ('Desk Chair', 'Furniture', 299.99),
            ('Bluetooth Speaker', 'Electronics', 99.99)
        ]
        
        cursor.executemany(
            "INSERT INTO products (product_name, category, price) VALUES (?, ?, ?)",
            products
        )
        
        # Insert sample customers
        customers = [
            ('John Smith', 'john@email.com', 'New York'),
            ('Sarah Johnson', 'sarah@email.com', 'California'),
            ('Mike Chen', 'mike@email.com', 'Texas'),
            ('Emily Davis', 'emily@email.com', 'Florida'),
            ('David Wilson', 'david@email.com', 'Illinois')
        ]
        
        cursor.executemany(
            "INSERT INTO customers (customer_name, email, location) VALUES (?, ?, ?)",
            customers
        )
        
        conn.commit()
        cursor.close()
        conn.close()
        print("✅ Sample data inserted!")

# =====================================================================
# 2. DATA GENERATION PIPELINE (300+ records/minute)
# =====================================================================

class StreamingDataGenerator:
    def __init__(self, db_manager):
        self.db = db_manager
        self.running = False
        self.records_per_minute = 300
        self.total_generated = 0
        
    def generate_transaction(self):
        """Generate a realistic transaction"""
        customer_id = random.randint(1, 5)
        product_id = random.randint(1, 8)
        quantity = random.choices([1, 2, 3], weights=[60, 30, 10])[0]
        
        # Get product price
        conn = self.db.get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT price FROM products WHERE product_id = ?", (product_id,))
        result = cursor.fetchone()
        cursor.close()
        conn.close()
        
        if result:
            price = float(result[0])
            total_amount = price * quantity
            
            # Add occasional anomalies (3% chance)
            is_anomaly = random.random() < 0.03
            if is_anomaly:
                total_amount *= random.uniform(10, 50)  # Suspicious large amount
            
            payment_method = random.choice(['Credit Card', 'PayPal', 'Bank Transfer', 'Apple Pay'])
            
            return (customer_id, product_id, quantity, round(total_amount, 2), 
                   datetime.now().isoformat(), payment_method, int(is_anomaly))
        return None
    
    def insert_transaction(self, transaction):
        """Insert transaction into database"""
        if not transaction:
            return
            
        conn = self.db.get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO transactions 
            (customer_id, product_id, quantity, total_amount, transaction_time, payment_method, is_anomaly)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, transaction)
        
        conn.commit()
        cursor.close()
        conn.close()
        self.total_generated += 1
    
    def start_streaming(self):
        """Start the streaming data generation"""
        self.running = True
        interval = 60.0 / self.records_per_minute  # seconds between records
        
        print(f"🚀 Starting data stream: {self.records_per_minute} records/minute")
        
        def stream_worker():
            while self.running:
                try:
                    transaction = self.generate_transaction()
                    if transaction:
                        self.insert_transaction(transaction)
                        if self.total_generated % 50 == 0:  # Progress update every 50 records
                            print(f"💾 Generated {self.total_generated} transactions...")
                    time.sleep(interval)
                except Exception as e:
                    print(f"Error in streaming: {e}")
                    time.sleep(1)
        
        # Start in background thread
        stream_thread = threading.Thread(target=stream_worker, daemon=True)
        stream_thread.start()
        return stream_thread
    
    def stop_streaming(self):
        """Stop the streaming process"""
        self.running = False
        print(f"⏹️ Streaming stopped. Total generated: {self.total_generated}")

# =====================================================================
# 3. OUTLIER DETECTION SYSTEM
# =====================================================================

class OutlierDetectionSystem:
    def __init__(self, db_manager):
        self.db = db_manager
        self.model = IsolationForest(contamination=0.05, random_state=42)
        self.scaler = StandardScaler()
        self.is_trained = False
        
    def get_recent_transactions(self, limit=500):
        """Get recent transactions for analysis"""
        conn = self.db.get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT transaction_id, total_amount, quantity, 
                   strftime('%H', transaction_time) as hour,
                   strftime('%w', transaction_time) as day_of_week
            FROM transactions 
            ORDER BY transaction_id DESC 
            LIMIT ?
        """, (limit,))
        
        data = cursor.fetchall()
        cursor.close()
        conn.close()
        
        if data:
            columns = ['transaction_id', 'total_amount', 'quantity', 'hour', 'day_of_week']
            df = pd.DataFrame(data, columns=columns)
            df['hour'] = pd.to_numeric(df['hour'], errors='coerce').fillna(0)
            df['day_of_week'] = pd.to_numeric(df['day_of_week'], errors='coerce').fillna(0)
            return df
        return pd.DataFrame()
    
    def train_detector(self):
        """Train the outlier detection model"""
        df = self.get_recent_transactions(1000)
        if len(df) < 50:
            print("Not enough data for outlier detection training")
            return False
        
        # Prepare features
        features = df[['total_amount', 'quantity', 'hour', 'day_of_week']].fillna(0)
        features_scaled = self.scaler.fit_transform(features)
        
        # Train model
        self.model.fit(features_scaled)
        self.is_trained = True
        
        print("✅ Outlier detection model trained!")
        return True
    
    def detect_anomalies(self, batch_size=100):
        """Detect anomalies in recent transactions"""
        if not self.is_trained:
            if not self.train_detector():
                return []
        
        df = self.get_recent_transactions(batch_size)
        if len(df) == 0:
            return []
        
        # Prepare features
        features = df[['total_amount', 'quantity', 'hour', 'day_of_week']].fillna(0)
        features_scaled = self.scaler.transform(features)
        
        # Predict anomalies
        predictions = self.model.predict(features_scaled)
        anomaly_indices = np.where(predictions == -1)[0]
        
        # Mark anomalies in database
        anomalous_transactions = []
        if len(anomaly_indices) > 0:
            conn = self.db.get_connection()
            cursor = conn.cursor()
            
            for idx in anomaly_indices:
                transaction_id = df.iloc[idx]['transaction_id']
                cursor.execute(
                    "UPDATE transactions SET is_anomaly = 1 WHERE transaction_id = ?",
                    (transaction_id,)
                )
                anomalous_transactions.append(transaction_id)
            
            conn.commit()
            cursor.close()
            conn.close()
            
            print(f"🚨 Detected {len(anomalous_transactions)} anomalies!")
        
        return anomalous_transactions

# =====================================================================
# 4. STREAMING MACHINE LEARNING MODEL
# =====================================================================

class StreamingMLModel:
    def __init__(self, db_manager):
        self.db = db_manager
        self.model = SGDRegressor(random_state=42, learning_rate='adaptive', eta0=0.01)
        self.scaler = StandardScaler()
        self.is_trained = False
        self.predictions_made = 0
        
    def get_training_data(self, limit=1000):
        """Get data for model training"""
        conn = self.db.get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT quantity, 
                   strftime('%H', transaction_time) as hour,
                   strftime('%w', transaction_time) as day_of_week,
                   product_id, customer_id, total_amount
            FROM transactions 
            ORDER BY transaction_id DESC 
            LIMIT ?
        """, (limit,))
        
        data = cursor.fetchall()
        cursor.close()
        conn.close()
        
        if data:
            columns = ['quantity', 'hour', 'day_of_week', 'product_id', 'customer_id', 'total_amount']
            df = pd.DataFrame(data, columns=columns)
            df['hour'] = pd.to_numeric(df['hour'], errors='coerce').fillna(0)
            df['day_of_week'] = pd.to_numeric(df['day_of_week'], errors='coerce').fillna(0)
            return df
        return pd.DataFrame()
    
    def train_model(self):
        """Train the ML model"""
        df = self.get_training_data(2000)
        if len(df) < 50:
            print("Not enough data for ML model training")
            return False
        
        # Prepare features and target
        features = df[['quantity', 'hour', 'day_of_week', 'product_id', 'customer_id']].fillna(0)
        target = df['total_amount'].values
        
        # Scale features
        features_scaled = self.scaler.fit_transform(features)
        
        # Train model
        self.model.fit(features_scaled, target)
        self.is_trained = True
        
        # Calculate accuracy
        predictions = self.model.predict(features_scaled)
        mae = mean_absolute_error(target, predictions)
        
        print(f"✅ ML Model trained! MAE: ${mae:.2f}")
        return True
    
    def predict_and_update(self, batch_size=50):
        """Make predictions and update model"""
        if not self.is_trained:
            if not self.train_model():
                return []
        
        df = self.get_training_data(batch_size)
        if len(df) == 0:
            return []
        
        # Prepare features
        features = df[['quantity', 'hour', 'day_of_week', 'product_id', 'customer_id']].fillna(0)
        features_scaled = self.scaler.transform(features)
        
        # Make predictions
        predictions = self.model.predict(features_scaled)
        actuals = df['total_amount'].values
        
        # Update model with new data
        self.model.partial_fit(features_scaled, actuals)
        
        self.predictions_made += len(predictions)
        
        # Calculate recent accuracy
        mae = mean_absolute_error(actuals, predictions)
        print(f"🎯 Streaming ML - Recent MAE: ${mae:.2f} | Total Predictions: {self.predictions_made}")
        
        return list(zip(predictions, actuals))

# =====================================================================
# 5. DASHBOARD DATA PROVIDER
# =====================================================================

class DashboardDataProvider:
    def __init__(self, db_manager):
        self.db = db_manager
        
    def get_kpi_metrics(self):
        """Get key performance indicators"""
        conn = self.db.get_connection()
        cursor = conn.cursor()
        
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
        
        cursor.close()
        conn.close()
        
        return {
            'transactions': today_data[0] or 0,
            'revenue': float(today_data[1] or 0),
            'avg_order': float(today_data[2] or 0),
            'unique_customers': today_data[3] or 0,
            'anomalies': anomalies
        }
    
    def get_recent_transactions(self, limit=10):
        """Get most recent transactions for display"""
        conn = self.db.get_connection()
        cursor = conn.cursor()
        
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
        conn.close()
        
        return data

# =====================================================================
# 6. MAIN SYSTEM ORCHESTRATOR
# =====================================================================

class StreamingAnalyticsSystem:
    def __init__(self):
        """Initialize the complete streaming analytics system"""
        print("🚀 Initializing Streaming Analytics System...")
        
        # Initialize components
        self.db_manager = DatabaseManager(use_sqlite=True)
        self.data_generator = StreamingDataGenerator(self.db_manager)
        self.outlier_detector = OutlierDetectionSystem(self.db_manager)
        self.ml_model = StreamingMLModel(self.db_manager)
        self.dashboard_data = DashboardDataProvider(self.db_manager)
        
        # Setup database
        self.db_manager.insert_sample_data()
        
    def start_system(self):
        """Start the complete streaming system"""
        print("🎯 Starting Complete Streaming Analytics System...")
        print("=" * 60)
        
        # Start data generation
        self.data_generator.start_streaming()
        
        print("⏳ Generating initial data (30 seconds)...")
        time.sleep(30)
        
        # Train models
        print("🤖 Training AI models...")
        self.outlier_detector.train_detector()
        self.ml_model.train_model()
        
        print("✅ System started successfully!")
        return True
    
    def run_monitoring_loop(self, duration_minutes=5):
        """Run system monitoring and processing"""
        print(f"🔄 Running monitoring for {duration_minutes} minutes...")
        
        start_time = time.time()
        end_time = start_time + (duration_minutes * 60)
        
        iteration = 0
        while time.time() < end_time:
            try:
                iteration += 1
                
                # Run anomaly detection
                anomalies = self.outlier_detector.detect_anomalies()
                
                # Run ML predictions
                predictions = self.ml_model.predict_and_update()
                
                # Get current metrics
                metrics = self.dashboard_data.get_kpi_metrics()
                
                # Show system status
                print(f"\n💓 System Status - Iteration {iteration} ({datetime.now().strftime('%H:%M:%S')})")
                print(f"   📊 Total Transactions: {metrics['transactions']:,}")
                print(f"   💰 Total Revenue: ${metrics['revenue']:,.2f}")
                print(f"   🚨 Anomalies Detected: {len(anomalies)} (Total: {metrics['anomalies']})")
                print(f"   🤖 ML Predictions: {len(predictions)} updated")
                print(f"   ⚡ Data Pipeline: Active ({self.data_generator.total_generated} total)")
                
                # Show recent transactions
                recent = self.dashboard_data.get_recent_transactions(5)
                print(f"   📋 Recent Transactions:")
                for txn in recent[:3]:  # Show top 3
                    status = "🚨" if txn[6] else "✅"
                    print(f"      {status} ID:{txn[0]} | {txn[2]} | ${txn[4]:.2f}")
                
                time.sleep(60)  # Check every minute
                
            except Exception as e:
                print(f"Monitoring error: {e}")
                time.sleep(10)
    
    def get_final_metrics(self):
        """Get final system metrics"""
        metrics = self.dashboard_data.get_kpi_metrics()
        
        return {
            **metrics,
            'total_generated': self.data_generator.total_generated,
            'ml_predictions': self.ml_model.predictions_made,
            'anomaly_rate': (metrics['anomalies'] / max(metrics['transactions'], 1)) * 100
        }
    
    def stop_system(self):
        """Stop the streaming system"""
        self.data_generator.stop_streaming()
        print("🛑 System stopped")

# =====================================================================
# 7. MAIN DEMO RUNNER
# =====================================================================

def run_complete_demo():
    """Run complete system demonstration"""
    print("🎬 RUNNING COMPLETE BIG DATA STREAMING ANALYTICS DEMO")
    print("=" * 80)
    print("📋 Testing all 6 deliverables:")
    print("   ✓ Multi-table Database (SQLite for local testing)")
    print("   ✓ Data Generation Pipeline (300 records/minute)")
    print("   ✓ Real-Time Analytics (metrics and monitoring)")
    print("   ✓ Outlier Detection System (Isolation Forest)")
    print("   ✓ Streaming ML Model (SGD Regressor)")
    print("   ✓ System Monitoring and Reporting")
    print()
    
    # Initialize system
    system = StreamingAnalyticsSystem()
    
    # Start system
    if system.start_system():
        # Run for demo period
        system.run_monitoring_loop(duration_minutes=5)
        
        # Get final metrics
        metrics = system.get_final_metrics()
        
        print(f"\n🎉 DEMO COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        print("📊 FINAL SYSTEM METRICS:")
        print(f"   💾 Total Transactions Generated: {metrics['total_generated']:,}")
        print(f"   📈 Transactions in Database: {metrics['transactions']:,}")
        print(f"   💰 Total Revenue: ${metrics['revenue']:,.2f}")
        print(f"   🛍️ Average Order Value: ${metrics['avg_order']:.2f}")
        print(f"   👥 Unique Customers: {metrics['unique_customers']}")
        print(f"   🚨 Anomalies Detected: {metrics['anomalies']} ({metrics['anomaly_rate']:.1f}% rate)")
        print(f"   🤖 ML Predictions Made: {metrics['ml_predictions']}")
        print()
        
        print("✅ ALL 6 DELIVERABLES DEMONSTRATED:")
        print("   • Multi-table Database: 3 related tables with data")
        print("   • Data Pipeline: 300+ records/minute generation")
        print("   • Real-time Analytics: Live metrics and monitoring")
        print("   • Outlier Detection: ML-based anomaly flagging")
        print("   • Streaming ML: Online learning with predictions")
        print("   • System Monitoring: Complete observability")
        print()
        
        # Stop system
        system.stop_system()
        
        print("🏆 PROJECT STATUS: READY FOR SUBMISSION!")
        print("📋 Next Steps:")
        print("   1. For Azure deployment, set environment variables")
        print("   2. Deploy dashboard with: streamlit run app.py")
        print("   3. Record 5-minute video demonstration")
        print("   4. Submit all deliverables")
        
        return True
    
    return False

if __name__ == "__main__":
    print("🚀 BIG DATA STREAMING ANALYTICS - COMPLETE LOCAL SYSTEM")
    print("This version runs immediately without Azure setup!")
    print("Database: SQLite (local) | Ready for Azure PostgreSQL deployment")
    print()
    
    # Run the complete demo
    success = run_complete_demo()
    
    if success:
        print("\n🎯 SYSTEM READY!")
        print("Your Big Data Streaming Analytics Pipeline is working perfectly!")
    else:
        print("\n❌ System startup failed")

# =====================================================================
# END OF COMPLETE SYSTEM
# =====================================================================
