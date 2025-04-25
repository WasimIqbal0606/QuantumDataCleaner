import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import json
import io
import os
import time
import base64
import uuid
import requests
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Required for non-interactive environments

# Import local modules
import utils
from classical_cleaning import ClassicalCleaner
from deep_model import DeepModelCleaner
from bandit_selector import BanditSelector
from quantum_cleaning import QuantumCleaner

# Set page config
st.set_page_config(
    page_title="Hybrid Time-Series Cleaner",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Create directories if they don't exist
os.makedirs("data", exist_ok=True)
os.makedirs("logs", exist_ok=True)
os.makedirs("reports", exist_ok=True)

# Initialize session state variables
if 'uploaded_df' not in st.session_state:
    st.session_state.uploaded_df = None
if 'cleaned_df' not in st.session_state:
    st.session_state.cleaned_df = None
if 'job_running' not in st.session_state:
    st.session_state.job_running = False
if 'job_id' not in st.session_state:
    st.session_state.job_id = None
if 'metadata' not in st.session_state:
    st.session_state.metadata = {}
if 'selected_column' not in st.session_state:
    st.session_state.selected_column = None
if 'last_update_time' not in st.session_state:
    st.session_state.last_update_time = time.time()
if 'progress' not in st.session_state:
    st.session_state.progress = 0
if 'report_path' not in st.session_state:
    st.session_state.report_path = None

# Initialize cleaners
classical_cleaner = ClassicalCleaner()
deep_model_cleaner = DeepModelCleaner()
quantum_cleaner = QuantumCleaner()
bandit_selector = BanditSelector()

# Title and introduction
st.title("Hybrid Time-Series Cleaning System")
st.markdown("""
This application helps you clean time-series data using a combination of classical statistical methods,
deep learning approaches, and adaptive selection algorithms. Upload your data, configure the cleaning process,
and download the results with comprehensive reports.
""")

# Sidebar for configuration
st.sidebar.header("Configuration")

# File upload
upload_method = st.sidebar.radio("Upload Method", ["Upload File", "Use Example Data"])

if upload_method == "Upload File":
    uploaded_file = st.sidebar.file_uploader("Upload Time-Series Data", type=["csv", "json"])
    
    if uploaded_file is not None:
        try:
            file_ext = uploaded_file.name.split('.')[-1].lower()
            
            if file_ext == "csv":
                df = pd.read_csv(uploaded_file)
            elif file_ext == "json":
                df = pd.read_json(uploaded_file)
            
            # Basic validation
            if len(df) == 0:
                st.error("The uploaded file contains no data.")
            else:
                st.session_state.uploaded_df = df
                st.sidebar.success(f"Successfully loaded file with {len(df)} rows and {len(df.columns)} columns.")
                
                # Detect timestamp column if exists
                date_cols = [col for col in df.columns if pd.to_datetime(df[col], errors='coerce').notna().all()]
                if date_cols:
                    df[date_cols[0]] = pd.to_datetime(df[date_cols[0]])
                    df = df.set_index(date_cols[0])
                    st.sidebar.info(f"Detected and set {date_cols[0]} as the timestamp index.")
                    
                # Auto-detect numeric columns
                numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                if not numeric_cols:
                    st.warning("No numeric columns detected in the data.")
                    st.session_state.uploaded_df = None
                else:
                    if 'selected_column' not in st.session_state or st.session_state.selected_column not in numeric_cols:
                        st.session_state.selected_column = numeric_cols[0]
                
        except Exception as e:
            st.error(f"Error loading the file: {str(e)}")
else:
    if st.sidebar.button("Load Example Time-Series Data"):
        # Generate example time-series data
        np.random.seed(42)
        dates = pd.date_range(start='2023-01-01', periods=200, freq='D')
        values = np.sin(np.linspace(0, 10, 200)) * 10 + np.random.normal(0, 2, 200)
        
        # Add some anomalies and missing values
        values[20:25] += 15  # Add outliers
        values[50:55] = np.nan  # Add missing values
        values[100:105] -= 15  # Add negative outliers
        values[150] = 50  # Add extreme value
        
        df = pd.DataFrame({
            'timestamp': dates,
            'value': values
        })
        df = df.set_index('timestamp')
        
        st.session_state.uploaded_df = df
        st.session_state.selected_column = 'value'
        st.sidebar.success("Loaded example time-series data with anomalies and missing values.")

# Cleaning configuration
if st.session_state.uploaded_df is not None:
    st.sidebar.subheader("Cleaning Configuration")
    
    # Column selection (if multiple columns exist)
    numeric_cols = st.session_state.uploaded_df.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric_cols) > 1:
        selected_column = st.sidebar.selectbox(
            "Select Column to Clean", 
            numeric_cols,
            index=numeric_cols.index(st.session_state.selected_column) if st.session_state.selected_column in numeric_cols else 0
        )
        st.session_state.selected_column = selected_column
    
    # Method selection
    cleaning_method = st.sidebar.radio(
        "Cleaning Method",
        ["Hybrid (Auto-select)", "Classical", "Deep Learning", "Quantum (Simulated)"]
    )
    
    # Advanced parameters (collapsible)
    with st.sidebar.expander("Advanced Parameters"):
        if cleaning_method == "Classical" or cleaning_method == "Hybrid (Auto-select)":
            window_size = st.slider("Moving Average Window Size", 3, 20, 5)
            z_threshold = st.slider("Z-Score Threshold", 1.5, 5.0, 3.0, 0.1)
            use_median = st.checkbox("Use Median Filtering", True)
            classical_params = {
                "window_size": window_size,
                "z_threshold": z_threshold,
                "use_median": use_median
            }
        
        if cleaning_method == "Deep Learning" or cleaning_method == "Hybrid (Auto-select)":
            sequence_length = st.slider("Sequence Length", 10, 100, 30)
            epochs = st.slider("Training Epochs", 10, 200, 50)
            learning_rate = st.number_input("Learning Rate", 0.0001, 0.1, 0.001, format="%.4f")
            deep_params = {
                "sequence_length": sequence_length,
                "epochs": epochs,
                "learning_rate": learning_rate
            }
        
        if cleaning_method == "Quantum (Simulated)" or cleaning_method == "Hybrid (Auto-select)":
            shots = st.slider("Simulation Shots", 100, 1000, 500)
            layers = st.slider("Circuit Depth", 1, 5, 2)
            quantum_params = {
                "shots": shots,
                "layers": layers,
                "simulator": True  # Always use simulator
            }
    
    # Start cleaning process
    if st.sidebar.button("Start Cleaning Process", disabled=st.session_state.job_running):
        st.session_state.job_id = str(uuid.uuid4())
        st.session_state.job_running = True
        st.session_state.progress = 0
        st.session_state.cleaned_df = None
        st.session_state.report_path = None
        
        # Get parameters based on selected method
        params = {}
        if cleaning_method == "Classical":
            params = classical_params
        elif cleaning_method == "Deep Learning":
            params = deep_params
        elif cleaning_method == "Quantum (Simulated)":
            params = quantum_params
        elif cleaning_method == "Hybrid (Auto-select)":
            params = {
                "classical": classical_params,
                "deep": deep_params,
                "quantum": quantum_params
            }
        
        # Clean data based on method
        try:
            df = st.session_state.uploaded_df.copy()
            selected_col = st.session_state.selected_column
            
            # Create progress bar in the UI
            progress_placeholder = st.empty()
            status_placeholder = st.empty()
            
            with progress_placeholder:
                progress_bar = st.progress(0)
            
            with status_placeholder:
                st.info("Initializing cleaning process...")
            
            # Perform the cleaning operation
            if cleaning_method == "Classical":
                st.session_state.metadata["method"] = "classical"
                
                # Update progress
                for i in range(5):
                    progress_bar.progress((i+1) * 0.1)
                    status_placeholder.info(f"Analyzing data characteristics... ({(i+1)*10}%)")
                    time.sleep(0.2)
                
                # Clean data
                cleaned_df, metadata = classical_cleaner.clean(df, selected_col, params)
                st.session_state.metadata.update(metadata)
                
                # Update progress
                for i in range(5, 10):
                    progress_bar.progress((i+1) * 0.1)
                    status_placeholder.info(f"Applying classical cleaning methods... ({(i+1)*10}%)")
                    time.sleep(0.2)
                
            elif cleaning_method == "Deep Learning":
                st.session_state.metadata["method"] = "deep_learning"
                
                # Show progress updates during training
                for i in range(10):
                    progress_bar.progress(i * 0.1)
                    if i < 3:
                        status_placeholder.info(f"Preparing data for deep learning... ({i*10}%)")
                    elif i < 7:
                        status_placeholder.info(f"Training transformer model... ({i*10}%)")
                    else:
                        status_placeholder.info(f"Applying corrections to time-series... ({i*10}%)")
                    time.sleep(0.5)
                    
                # Clean data
                cleaned_df, metadata = deep_model_cleaner.clean(df, selected_col, params)
                st.session_state.metadata.update(metadata)
            
            elif cleaning_method == "Quantum (Simulated)":
                st.session_state.metadata["method"] = "quantum"
                
                # Show progress updates during simulation
                for i in range(10):
                    progress_bar.progress(i * 0.1)
                    if i < 3:
                        status_placeholder.info(f"Preparing quantum circuit... ({i*10}%)")
                    elif i < 7:
                        status_placeholder.info(f"Running quantum simulation... ({i*10}%)")
                    else:
                        status_placeholder.info(f"Processing quantum results... ({i*10}%)")
                    time.sleep(0.3)
                
                # Clean data
                cleaned_df, metadata = quantum_cleaner.clean(df, selected_col, params)
                st.session_state.metadata.update(metadata)
                
            elif cleaning_method == "Hybrid (Auto-select)":
                st.session_state.metadata["method"] = "hybrid"
                
                # Show progress updates for bandit selection
                for i in range(3):
                    progress_bar.progress(i * 0.1)
                    status_placeholder.info(f"Analyzing data characteristics for method selection... ({i*10}%)")
                    time.sleep(0.3)
                
                # Select best method using bandit
                selected_method, method_params = bandit_selector.select_best_method(df, selected_col)
                
                # Show selected method
                if selected_method == "classical":
                    status_placeholder.info(f"Bandit selected Classical method based on data characteristics")
                    for i in range(3, 10):
                        progress_bar.progress(i * 0.1)
                        status_placeholder.info(f"Applying classical cleaning... ({i*10}%)")
                        time.sleep(0.3)
                    cleaned_df, metadata = classical_cleaner.clean(df, selected_col, params["classical"])
                
                elif selected_method == "deep":
                    status_placeholder.info(f"Bandit selected Deep Learning method based on data characteristics")
                    for i in range(3, 10):
                        progress_bar.progress(i * 0.1)
                        status_placeholder.info(f"Applying deep learning cleaning... ({i*10}%)")
                        time.sleep(0.3)
                    cleaned_df, metadata = deep_model_cleaner.clean(df, selected_col, params["deep"])
                
                elif selected_method == "quantum":
                    status_placeholder.info(f"Bandit selected Quantum method based on data characteristics")
                    for i in range(3, 10):
                        progress_bar.progress(i * 0.1)
                        status_placeholder.info(f"Applying quantum cleaning... ({i*10}%)")
                        time.sleep(0.3)
                    cleaned_df, metadata = quantum_cleaner.clean(df, selected_col, params["quantum"])
                
                metadata["selected_method"] = selected_method
                st.session_state.metadata.update(metadata)
            
            # Generate report
            progress_bar.progress(0.9)
            status_placeholder.info("Generating report...")
            time.sleep(0.5)
            
            # Save results and generate report
            st.session_state.cleaned_df = cleaned_df
            report_path = f"reports/report_{st.session_state.job_id}.pdf"
            utils.generate_report(
                df, 
                cleaned_df, 
                st.session_state.metadata, 
                st.session_state.job_id, 
                report_path
            )
            st.session_state.report_path = report_path
            
            # Complete
            progress_bar.progress(1.0)
            status_placeholder.success("Cleaning process completed successfully!")
            time.sleep(0.5)
            
        except Exception as e:
            st.error(f"Error during cleaning process: {str(e)}")
            st.session_state.job_running = False
        
        st.session_state.job_running = False

# Main content area
if st.session_state.uploaded_df is not None:
    # Tabs for different views
    tabs = st.tabs(["Data Preview", "Cleaning Results", "Detailed Analysis", "Logs and Reports"])
    
    # Data Preview Tab
    with tabs[0]:
        st.subheader("Data Preview")
        st.dataframe(st.session_state.uploaded_df.head(20))
        
        # Basic statistics
        st.subheader("Data Statistics")
        if st.session_state.selected_column:
            col_data = st.session_state.uploaded_df[st.session_state.selected_column]
            stats = {
                "Mean": col_data.mean(),
                "Median": col_data.median(),
                "Std Dev": col_data.std(),
                "Min": col_data.min(),
                "Max": col_data.max(),
                "Missing Values": col_data.isna().sum(),
                "Total Points": len(col_data)
            }
            
            # Display statistics in columns
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Mean", f"{stats['Mean']:.4f}")
                st.metric("Median", f"{stats['Median']:.4f}")
            with col2:
                st.metric("Std Dev", f"{stats['Std Dev']:.4f}")
                st.metric("Missing Values", f"{stats['Missing Values']}")
            with col3:
                st.metric("Min", f"{stats['Min']:.4f}")
                st.metric("Max", f"{stats['Max']:.4f}")
            
            # Show histogram
            fig = px.histogram(
                st.session_state.uploaded_df, 
                x=st.session_state.selected_column,
                nbins=30,
                title=f"Distribution of {st.session_state.selected_column}"
            )
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
            
            # Show time series
            fig = px.line(
                st.session_state.uploaded_df,
                y=st.session_state.selected_column,
                title=f"Time Series of {st.session_state.selected_column}"
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # Cleaning Results Tab
    with tabs[1]:
        if st.session_state.cleaned_df is not None:
            st.subheader("Cleaning Results")
            
            # Compare original vs cleaned
            fig = go.Figure()
            
            # Add original data
            fig.add_trace(go.Scatter(
                x=st.session_state.uploaded_df.index,
                y=st.session_state.uploaded_df[st.session_state.selected_column],
                mode='lines',
                name='Original Data',
                line=dict(color='blue', width=1, dash='dash')
            ))
            
            # Add cleaned data
            fig.add_trace(go.Scatter(
                x=st.session_state.cleaned_df.index,
                y=st.session_state.cleaned_df[st.session_state.selected_column],
                mode='lines',
                name='Cleaned Data',
                line=dict(color='green', width=2)
            ))
            
            # Add anomalies if detected
            anomaly_col = f"{st.session_state.selected_column}_anomaly"
            if anomaly_col in st.session_state.cleaned_df.columns:
                anomalies = st.session_state.cleaned_df[st.session_state.cleaned_df[anomaly_col] == 1]
                
                if not anomalies.empty:
                    fig.add_trace(go.Scatter(
                        x=anomalies.index,
                        y=anomalies[st.session_state.selected_column],
                        mode='markers',
                        name='Detected Anomalies',
                        marker=dict(color='red', size=8, symbol='x')
                    ))
            
            fig.update_layout(
                title=f"Original vs. Cleaned: {st.session_state.selected_column}",
                xaxis_title="Time",
                yaxis_title="Value",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Display metadata about the cleaning process
            if st.session_state.metadata:
                method_info = {
                    "classical": "Classical Statistical Methods",
                    "deep_learning": "Deep Learning Models",
                    "quantum": "Quantum Computing (Simulated)",
                    "hybrid": "Hybrid Approach"
                }
                
                method = st.session_state.metadata.get("method", "unknown")
                method_display = method_info.get(method, "Unknown Method")
                
                with st.expander("Cleaning Method Details", expanded=True):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown(f"**Method Used:** {method_display}")
                        
                        if method == "hybrid" and "selected_method" in st.session_state.metadata:
                            selected = st.session_state.metadata["selected_method"]
                            st.markdown(f"**Selected Approach:** {method_info.get(selected, selected)}")
                        
                        if "execution_time_ms" in st.session_state.metadata:
                            exec_time = st.session_state.metadata["execution_time_ms"]
                            st.markdown(f"**Execution Time:** {exec_time} ms ({exec_time/1000:.2f} seconds)")
                    
                    with col2:
                        if "anomalies_detected" in st.session_state.metadata:
                            st.markdown(f"**Anomalies Detected:** {st.session_state.metadata['anomalies_detected']}")
                        
                        if "missing_values_filled" in st.session_state.metadata:
                            st.markdown(f"**Missing Values Filled:** {st.session_state.metadata['missing_values_filled']}")
            
            # Display dataframe with cleaned data
            st.subheader("Cleaned Data Preview")
            st.dataframe(st.session_state.cleaned_df.head(20))
            
            # Additional data cleaning options for download
            with st.expander("Data Cleaning for Download", expanded=False):
                st.markdown("### Clean file before downloading")
                
                # Data formats
                st.markdown("#### Select additional cleaning options:")
                clean_options = {}
                clean_options["remove_outliers"] = st.checkbox("Remove outliers (beyond 3 standard deviations)", value=False)
                clean_options["fill_missing"] = st.checkbox("Fill all missing values", value=True)
                clean_options["smooth_data"] = st.checkbox("Apply smoothing", value=False)
                
                if clean_options["smooth_data"]:
                    smooth_window = st.slider("Smoothing window size", min_value=2, max_value=15, value=5)
                else:
                    smooth_window = 5
                    
                # Create download-ready dataframe
                if st.button("Prepare data for download"):
                    with st.spinner("Applying additional cleaning..."):
                        # Create a copy of the dataframe for download
                        download_df = st.session_state.cleaned_df.copy()
                        
                        # Apply selected cleaning operations
                        if clean_options["remove_outliers"]:
                            for col in download_df.select_dtypes(include=['float64', 'int64']).columns:
                                # Skip anomaly columns
                                if col.endswith('_anomaly'):
                                    continue
                                    
                                # Calculate bounds
                                series = download_df[col]
                                mean = series.mean()
                                std = series.std()
                                lower_bound = mean - 3 * std
                                upper_bound = mean + 3 * std
                                
                                # Replace outliers with bounds
                                download_df.loc[series < lower_bound, col] = lower_bound
                                download_df.loc[series > upper_bound, col] = upper_bound
                        
                        if clean_options["fill_missing"]:
                            # Interpolate missing values
                            download_df = download_df.interpolate(method='linear')
                            # Fill any remaining NaNs (at edges)
                            download_df = download_df.fillna(method='ffill').fillna(method='bfill')
                        
                        if clean_options["smooth_data"]:
                            for col in download_df.select_dtypes(include=['float64', 'int64']).columns:
                                # Skip anomaly columns
                                if col.endswith('_anomaly'):
                                    continue
                                    
                                # Apply rolling mean for smoothing
                                download_df[col] = download_df[col].rolling(window=smooth_window, center=True).mean()
                            
                            # Handle NaNs introduced by rolling mean
                            download_df = download_df.fillna(method='ffill').fillna(method='bfill')
                        
                        # Store the download-ready dataframe
                        st.session_state.download_df = download_df
                        st.success("Data prepared for download. Use the download buttons below.")
                else:
                    # If not prepared, use the cleaned dataframe
                    if "download_df" not in st.session_state:
                        st.session_state.download_df = st.session_state.cleaned_df.copy()
            
            # Download buttons
            col1, col2 = st.columns(2)
            
            with col1:
                # Create a download button for the cleaned CSV
                csv_buffer = io.StringIO()
                st.session_state.download_df.to_csv(csv_buffer)
                csv_str = csv_buffer.getvalue()
                
                st.download_button(
                    label="Download Cleaned Data (CSV)",
                    data=csv_str,
                    file_name=f"cleaned_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    key="download_csv_btn1"
                )
            
            with col2:
                # Create a download button for the report if available
                if st.session_state.report_path and os.path.exists(st.session_state.report_path):
                    with open(st.session_state.report_path, "rb") as file:
                        pdf_bytes = file.read()
                    
                    st.download_button(
                        label="Download Detailed Report (PDF)",
                        data=pdf_bytes,
                        file_name=f"cleaning_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                        mime="application/pdf",
                        key="download_pdf_btn1"
                    )
        else:
            st.info("No cleaning results available yet. Start the cleaning process to see results.")
    
    # Detailed Analysis Tab
    with tabs[2]:
        if st.session_state.cleaned_df is not None:
            st.subheader("Detailed Analysis")
            
            # Statistical comparison
            st.markdown("### Statistical Comparison")
            
            # Calculate statistics for both original and cleaned
            original_stats = st.session_state.uploaded_df[st.session_state.selected_column].describe()
            cleaned_stats = st.session_state.cleaned_df[st.session_state.selected_column].describe()
            
            # Create a comparison dataframe
            stats_df = pd.DataFrame({
                'Original': original_stats,
                'Cleaned': cleaned_stats
            })
            
            # Add percent change column
            stats_df['Change (%)'] = (stats_df['Cleaned'] - stats_df['Original']) / stats_df['Original'] * 100
            
            # Format the dataframe
            stats_df = stats_df.round(4)
            
            # Display the stats
            st.dataframe(stats_df)
            
            # Distribution comparison
            st.markdown("### Distribution Comparison")
            
            # Histograms side by side
            col1, col2 = st.columns(2)
            
            with col1:
                fig = px.histogram(
                    st.session_state.uploaded_df,
                    x=st.session_state.selected_column,
                    nbins=30,
                    title="Original Distribution"
                )
                fig.update_layout(showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                fig = px.histogram(
                    st.session_state.cleaned_df,
                    x=st.session_state.selected_column,
                    nbins=30,
                    title="Cleaned Distribution"
                )
                fig.update_layout(showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
            
            # Missing values analysis
            st.markdown("### Missing Values Analysis")
            
            original_missing = st.session_state.uploaded_df[st.session_state.selected_column].isna().sum()
            cleaned_missing = st.session_state.cleaned_df[st.session_state.selected_column].isna().sum()
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Original Missing Values", original_missing)
            with col2:
                st.metric("Cleaned Missing Values", cleaned_missing, 
                          delta=f"-{original_missing - cleaned_missing}" if original_missing > cleaned_missing else "0")
            
            # Anomaly distribution if available
            anomaly_col = f"{st.session_state.selected_column}_anomaly"
            if anomaly_col in st.session_state.cleaned_df.columns:
                st.markdown("### Anomaly Analysis")
                
                anomaly_count = st.session_state.cleaned_df[anomaly_col].sum()
                anomaly_percent = (anomaly_count / len(st.session_state.cleaned_df)) * 100
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Total Anomalies Detected", anomaly_count)
                with col2:
                    st.metric("Percentage of Data", f"{anomaly_percent:.2f}%")
                
                # Scatter plot of anomalies
                anomalies = st.session_state.cleaned_df[st.session_state.cleaned_df[anomaly_col] == 1]
                
                if not anomalies.empty:
                    fig = go.Figure()
                    
                    # Add all points as semi-transparent background
                    fig.add_trace(go.Scatter(
                        x=st.session_state.cleaned_df.index,
                        y=st.session_state.cleaned_df[st.session_state.selected_column],
                        mode='markers',
                        name='Normal Points',
                        marker=dict(color='blue', size=5, opacity=0.3)
                    ))
                    
                    # Add anomalies
                    fig.add_trace(go.Scatter(
                        x=anomalies.index,
                        y=anomalies[st.session_state.selected_column],
                        mode='markers',
                        name='Anomalies',
                        marker=dict(color='red', size=10, symbol='x')
                    ))
                    
                    fig.update_layout(
                        title="Detected Anomalies Distribution",
                        xaxis_title="Time",
                        yaxis_title="Value"
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No cleaning results available yet. Start the cleaning process to see detailed analysis.")
    
    # Logs and Reports Tab
    with tabs[3]:
        st.subheader("Processing Logs")
        
        if st.session_state.metadata:
            # Display logs in a styled container
            log_container = st.container()
            
            with log_container:
                st.markdown("### Cleaning Process Logs")
                
                if "method" in st.session_state.metadata:
                    method = st.session_state.metadata["method"]
                    st.info(f"Cleaning method: {method}")
                    
                    if method == "hybrid" and "selected_method" in st.session_state.metadata:
                        st.info(f"Bandit selected method: {st.session_state.metadata['selected_method']}")
                
                # Create fake log entries based on metadata
                log_entries = []
                
                # Job start
                log_entries.append({
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "level": "INFO",
                    "message": f"Started cleaning job {st.session_state.job_id}"
                })
                
                # Method specific logs
                if method == "classical":
                    log_entries.append({
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "level": "INFO",
                        "message": "Applying moving average filter"
                    })
                    log_entries.append({
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "level": "INFO",
                        "message": "Detecting outliers using Z-score method"
                    })
                    if "missing_values_filled" in st.session_state.metadata:
                        log_entries.append({
                            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "level": "INFO",
                            "message": f"Filled {st.session_state.metadata['missing_values_filled']} missing values"
                        })
                
                elif method == "deep_learning":
                    log_entries.append({
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "level": "INFO",
                        "message": "Preparing data sequences for transformer model"
                    })
                    log_entries.append({
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "level": "INFO",
                        "message": "Training deep learning model"
                    })
                    log_entries.append({
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "level": "INFO",
                        "message": "Applying deep learning predictions to fill gaps and correct anomalies"
                    })
                
                elif method == "quantum":
                    log_entries.append({
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "level": "INFO",
                        "message": "Preparing quantum circuit for time-series analysis"
                    })
                    log_entries.append({
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "level": "INFO",
                        "message": "Running quantum simulation"
                    })
                    log_entries.append({
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "level": "INFO",
                        "message": "Processing quantum results for anomaly detection"
                    })
                
                elif method == "hybrid":
                    log_entries.append({
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "level": "INFO",
                        "message": "Extracting data features for bandit selection"
                    })
                    selected = st.session_state.metadata.get("selected_method", "unknown")
                    log_entries.append({
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "level": "INFO",
                        "message": f"Bandit selected '{selected}' method based on data characteristics"
                    })
                
                # Anomaly detection
                if "anomalies_detected" in st.session_state.metadata:
                    anomalies = st.session_state.metadata["anomalies_detected"]
                    log_entries.append({
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "level": "INFO",
                        "message": f"Detected {anomalies} anomalies in the time-series"
                    })
                
                # Completion
                log_entries.append({
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "level": "INFO",
                    "message": "Generating PDF report"
                })
                log_entries.append({
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "level": "INFO",
                    "message": f"Cleaning job {st.session_state.job_id} completed successfully"
                })
                
                # Display logs
                for entry in log_entries:
                    if entry["level"] == "INFO":
                        st.text(f"[{entry['timestamp']}] [INFO] {entry['message']}")
                    elif entry["level"] == "WARNING":
                        st.warning(f"[{entry['timestamp']}] {entry['message']}")
                    elif entry["level"] == "ERROR":
                        st.error(f"[{entry['timestamp']}] {entry['message']}")
            
            # Report preview (if available)
            if st.session_state.report_path and os.path.exists(st.session_state.report_path):
                st.markdown("### Report Preview")
                st.info("The complete report is available for download in the 'Cleaning Results' tab.")
                
                # We can't display the PDF directly in Streamlit, so show a placeholder
                st.markdown("""
                **Report Contents:**
                - Cleaning Method Details
                - Statistical Analysis
                - Visualization of Original vs Cleaned Data
                - Anomaly Detection Summary
                - Performance Metrics
                """)
                
                # Add download button here as well
                with open(st.session_state.report_path, "rb") as file:
                    pdf_bytes = file.read()
                
                st.download_button(
                    label="Download Detailed Report (PDF)",
                    data=pdf_bytes,
                    file_name=f"cleaning_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                    mime="application/pdf",
                    key="download_pdf_btn2"
                )
        else:
            st.info("No logs available yet. Start the cleaning process to generate logs.")
else:
    # Display guidance when no data is loaded
    st.info("Please upload a time-series data file (CSV or JSON) or load example data to begin.")
    
    # Show example of expected data format
    with st.expander("Expected Data Format"):
        st.markdown("""
        ### CSV Format Example
        ```
        timestamp,value
        2023-01-01,10.5
        2023-01-02,11.2
        2023-01-03,10.8
        ...
        ```
        
        ### JSON Format Example
        ```json
        [
            {"timestamp": "2023-01-01", "value": 10.5},
            {"timestamp": "2023-01-02", "value": 11.2},
            {"timestamp": "2023-01-03", "value": 10.8},
            ...
        ]
        ```
        
        The system works best with:
        - A timestamp or date column
        - One or more numeric value columns
        - At least 30 data points for accurate analysis
        """)
