# 🚀 INSTANT RUN SCRIPT - Big Data Streaming Analytics
# Double-click this file or run: python instant_run.py

import subprocess
import sys
import os
import time
import threading

def print_header():
    print("🚀 BIG DATA STREAMING ANALYTICS - INSTANT RUNNER")
    print("=" * 60)
    print("This script will:")
    print("1. Start the data generation pipeline")
    print("2. Launch the Streamlit dashboard")
    print("3. Open your browser automatically")
    print()

def install_requirements():
    """Install required packages"""
    print("📦 Installing required packages...")
    packages = [
        'streamlit', 'pandas', 'numpy', 'plotly', 
        'scikit-learn', 'sqlite3'
    ]
    
    for package in packages:
        if package == 'sqlite3':
            continue  # Built into Python
        try:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', package], 
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except:
            print(f"⚠️ Could not install {package}")
    
    print("✅ Packages installed!")

def start_data_generator():
    """Start the data generation in background"""
    print("🔄 Starting data generation pipeline...")
    
    def run_generator():
        try:
            subprocess.run([sys.executable, 'complete_system.py'], check=True)
        except Exception as e:
            print(f"⚠️ Data generator error: {e}")
    
    # Start in background thread
    thread = threading.Thread(target=run_generator, daemon=True)
    thread.start()
    
    # Give it time to start
    time.sleep(3)
    return thread

def start_dashboard():
    """Start Streamlit dashboard"""
    print("📊 Starting Streamlit dashboard...")
    print("🌐 Dashboard will open at: http://localhost:8501")
    print()
    print("Press Ctrl+C to stop the system")
    print("=" * 60)
    
    try:
        subprocess.run([sys.executable, '-m', 'streamlit', 'run', 'local_dashboard.py',
                       '--server.port', '8501'], check=True)
    except KeyboardInterrupt:
        print("\n⏹️ System stopped by user")
    except Exception as e:
        print(f"❌ Dashboard error: {e}")

def main():
    print_header()
    
    # Check if required files exist
    required_files = ['complete_system.py', 'local_dashboard.py']
    missing_files = [f for f in required_files if not os.path.exists(f)]
    
    if missing_files:
        print(f"❌ Missing required files: {', '.join(missing_files)}")
        print("Please ensure all files are in the same directory.")
        input("Press Enter to exit...")
        return
    
    # Install packages
    install_requirements()
    
    # Start data generator
    generator_thread = start_data_generator()
    
    # Wait a bit for initial data generation
    print("⏳ Generating initial data (10 seconds)...")
    time.sleep(10)
    
    # Start dashboard (this will block until stopped)
    start_dashboard()

if __name__ == "__main__":
    main()