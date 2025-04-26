import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import io
import json
import os
import tempfile
import time
import uuid
from datetime import datetime, timedelta
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

# Import custom modules
from utils import ResultObject
from classical_cleaning import ClassicalCleaner
from deep_model import DeepModelCleaner
from quantum_cleaning import QuantumCleaner
from bandit_selector import BanditSelector
from database_service import DatabaseService
from ocr_extractor import OCRExtractor

# Initialize services
classical_cleaner = ClassicalCleaner()
deep_cleaner = DeepModelCleaner()
quantum_cleaner = QuantumCleaner()
bandit_selector = BanditSelector()
db_service = DatabaseService()
ocr_extractor = OCRExtractor()

# Configure page
st.set_page_config(
    page_title="Advanced Hybrid Time-Series Cleaning System",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state for persistent data
if "uploaded_df" not in st.session_state:
    st.session_state.uploaded_df = None
if "cleaned_df" not in st.session_state:
    st.session_state.cleaned_df = None
if "selected_column" not in st.session_state:
    st.session_state.selected_column = None
if "numeric_columns" not in st.session_state:
    st.session_state.numeric_columns = []
if "job_id" not in st.session_state:
    st.session_state.job_id = str(uuid.uuid4())
if "logs" not in st.session_state:
    st.session_state.logs = []
if "metadata" not in st.session_state:
    st.session_state.metadata = {}

# Initialize OCR-related session state
if "ocr_source_type" not in st.session_state:
    st.session_state.ocr_source_type = None
if "ocr_image" not in st.session_state:
    st.session_state.ocr_image = None
if "ocr_extracted_text" not in st.session_state:
    st.session_state.ocr_extracted_text = None
if "ocr_extracted_df" not in st.session_state:
    st.session_state.ocr_extracted_df = None
if "ocr_metadata" not in st.session_state:
    st.session_state.ocr_metadata = {}
if "ocr_use_quantum" not in st.session_state:
    st.session_state.ocr_use_quantum = False

# Function to add log entry
def add_log(level, message):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    st.session_state.logs.append({
        "timestamp": timestamp,
        "level": level,
        "message": message
    })

# Function to plot time series
def plot_time_series(df, column, title=None, height=400):
    fig = px.line(df, y=column, title=title or f"Time Series: {column}")
    fig.update_layout(height=height)
    return fig

# Helper function to get column statistics
def get_column_stats(df, column):
    series = df[column]
    stats = {
        "mean": series.mean(),
        "median": series.median(),
        "std": series.std(),
        "min": series.min(),
        "max": series.max(),
        "missing": series.isna().sum(),
        "missing_pct": (series.isna().sum() / len(series)) * 100
    }
    return stats

# Generate data quality report
def generate_quality_report(df, column):
    # Get basic statistics
    stats = get_column_stats(df, column)
    
    # Calculate additional metrics
    report = {
        "total_rows": len(df),
        "missing_values": int(stats["missing"]),
        "missing_percentage": float(stats["missing_pct"]),
        "outliers_z_score": 0,
        "outliers_iqr": 0,
        "potential_anomalies": 0,
        "stats": stats
    }
    
    # Detect outliers using Z-score method
    z_scores = np.abs((df[column] - stats["mean"]) / stats["std"])
    report["outliers_z_score"] = int((z_scores > 3).sum())
    
    # Detect outliers using IQR method
    q1 = df[column].quantile(0.25)
    q3 = df[column].quantile(0.75)
    iqr = q3 - q1
    report["outliers_iqr"] = int(((df[column] < (q1 - 1.5 * iqr)) | (df[column] > (q3 + 1.5 * iqr))).sum())
    
    # Check for sudden spikes or drops
    diffs = df[column].diff().abs()
    threshold = diffs.mean() + 3 * diffs.std()
    report["potential_anomalies"] = int((diffs > threshold).sum())
    
    return report

# Function to handle file upload
def process_uploaded_file(uploaded_file):
    try:
        # Check file extension
        file_extension = uploaded_file.name.split(".")[-1].lower()
        
        if file_extension == "csv":
            df = pd.read_csv(uploaded_file)
        elif file_extension in ["xls", "xlsx"]:
            df = pd.read_excel(uploaded_file)
        elif file_extension == "json":
            df = pd.read_json(uploaded_file)
        elif file_extension == "txt":
            # Try to parse as CSV with different delimiters
            try:
                df = pd.read_csv(uploaded_file, delimiter=",")
            except:
                try:
                    df = pd.read_csv(uploaded_file, delimiter="\t")
                except:
                    try:
                        df = pd.read_csv(uploaded_file, delimiter=";")
                    except:
                        raise Exception("Could not parse TXT file with common delimiters")
        else:
            raise Exception(f"Unsupported file format: {file_extension}")
        
        # Find numeric columns
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Store in session state
        st.session_state.uploaded_df = df
        st.session_state.numeric_columns = numeric_columns
        
        # Reset selected column if it's not in the new dataframe
        if st.session_state.selected_column not in numeric_columns:
            st.session_state.selected_column = None
        
        # Reset cleaned dataframe
        st.session_state.cleaned_df = None
        
        # Add log entry
        add_log("info", f"File uploaded successfully: {uploaded_file.name} with {len(df)} rows and {len(df.columns)} columns")
        
        return True, df, numeric_columns
        
    except Exception as e:
        add_log("error", f"Error processing file: {str(e)}")
        return False, None, []

# Function to perform OCR extraction
def perform_ocr_extraction(source, source_type, use_quantum=False):
    try:
        start_time = time.time()
        
        # Store the source type and quantum setting
        st.session_state.ocr_source_type = source_type
        st.session_state.ocr_use_quantum = use_quantum
        
        # If it's an image, store it for display
        if source_type == "image":
            st.session_state.ocr_image = source
            
        # Perform OCR extraction using the OCR service
        extracted_df, metadata = ocr_extractor.extract_time_series(
            source=source,
            source_type=source_type,
            use_quantum=use_quantum
        )
        
        # Calculate execution time
        execution_time_ms = (time.time() - start_time) * 1000
        
        # Extract text if available
        if "extracted_text" in metadata:
            st.session_state.ocr_extracted_text = metadata["extracted_text"]
        else:
            st.session_state.ocr_extracted_text = "No text extracted"
            
        # Store the extracted dataframe and metadata
        st.session_state.ocr_extracted_df = extracted_df
        
        # Add execution time to metadata
        metadata["execution_time_ms"] = execution_time_ms
        st.session_state.ocr_metadata = metadata
        
        # Add log entry
        add_log("info", f"OCR extraction completed in {execution_time_ms:.1f} ms for {source_type} source")
        
        return True
    except Exception as e:
        add_log("error", f"Error during OCR extraction: {str(e)}")
        st.session_state.ocr_extracted_text = f"Error: {str(e)}"
        st.session_state.ocr_extracted_df = None
        st.session_state.ocr_metadata = {"error": str(e)}
        return False

# Main application layout
st.title("Advanced Hybrid Time-Series Cleaning System")
st.markdown("""
This application provides comprehensive time-series data cleaning and analysis using a hybrid 
approach that combines classical statistical methods, deep learning, and quantum-inspired algorithms.
""")

# Sidebar for controls
with st.sidebar:
    st.header("Controls")
    
    upload_option = st.radio("Select Data Source", 
                           ["Upload Time-Series Data", "Load Example Data", "OCR Extract"])
    
    if upload_option == "Upload Time-Series Data":
        uploaded_file = st.file_uploader("Upload your time-series data", 
                                       type=["csv", "xlsx", "xls", "json", "txt"])
        
        if uploaded_file is not None:
            # Process the uploaded file
            success, _, _ = process_uploaded_file(uploaded_file)
            
            if not success:
                st.error("Failed to process the uploaded file. Check the logs for details.")
    elif upload_option == "OCR Extract":
        # OCR Extraction Options
        st.markdown("### OCR Extraction Options")
        
        ocr_source_type = st.radio("Select Source Type", ["Image", "PDF"])
        use_quantum = st.checkbox("Use Quantum-Inspired Enhancement", 
                                help="Applies quantum-inspired image enhancement for better OCR quality")
        
        if ocr_source_type == "Image":
            uploaded_image = st.file_uploader("Upload Image with Time-Series Data", 
                                            type=["jpg", "jpeg", "png", "bmp"])
            
            if uploaded_image is not None:
                # Process the uploaded image with OCR
                if st.button("Extract Data from Image"):
                    with st.spinner("Performing OCR extraction..."):
                        # Convert to bytes for processing
                        image_bytes = uploaded_image.getvalue()
                        success = perform_ocr_extraction(image_bytes, "image", use_quantum)
                        
                        if success:
                            st.success("OCR extraction completed! View results in the OCR Extraction tab.")
                        else:
                            st.error("OCR extraction failed. Check the logs for details.")
        
        elif ocr_source_type == "PDF":
            uploaded_pdf = st.file_uploader("Upload PDF with Time-Series Data", 
                                          type=["pdf"])
            
            if uploaded_pdf is not None:
                # Process the uploaded PDF with OCR
                if st.button("Extract Data from PDF"):
                    with st.spinner("Performing OCR extraction..."):
                        # Convert to bytes for processing
                        pdf_bytes = uploaded_pdf.getvalue()
                        success = perform_ocr_extraction(pdf_bytes, "pdf", use_quantum)
                        
                        if success:
                            st.success("OCR extraction completed! View results in the OCR Extraction tab.")
                        else:
                            st.error("OCR extraction failed. Check the logs for details.")
    else:  # Use Example Data
        if st.sidebar.button("Load Example Time-Series Data"):
            # Create example data
            dates = pd.date_range(start="2023-01-01", periods=365, freq="D")
            values = np.sin(np.linspace(0, 10, 365)) * 5 + np.random.normal(0, 0.5, 365)
            
            # Add some outliers
            values[50] = 15
            values[150] = -10
            values[250] = 20
            
            # Add missing values
            values[75:85] = np.nan
            values[200:205] = np.nan
            
            # Create dataframe
            df = pd.DataFrame({"date": dates, "value": values})
            
            # Store in session state
            st.session_state.uploaded_df = df
            st.session_state.numeric_columns = ["value"]
            st.session_state.selected_column = "value"
            
            # Reset cleaned dataframe
            st.session_state.cleaned_df = None
            
            # Add log entry
            add_log("info", "Loaded example time-series data with 365 data points")
            
            st.success("Example data loaded successfully!")
    
    # Column selection (only shown when data is loaded)
    if st.session_state.uploaded_df is not None:
        st.header("Data Cleaning")
        
        # Show column selection
        if st.session_state.numeric_columns:
            st.session_state.selected_column = st.selectbox(
                "Select column to clean", 
                options=st.session_state.numeric_columns,
                index=0 if st.session_state.selected_column is None else 
                      st.session_state.numeric_columns.index(st.session_state.selected_column)
            )
        else:
            st.warning("No numeric columns found in the data.")
        
        # Cleaning method selection
        st.subheader("Cleaning Method")
        cleaning_method = st.radio(
            "Select cleaning method",
            ["Classical Statistical Methods", "Deep Learning Based", 
             "Quantum-Inspired", "Hybrid (Auto-select)"]
        )
        
        # Method-specific parameters
        if cleaning_method == "Classical Statistical Methods":
            st.subheader("Classical Parameters")
            
            window_size = st.slider("Window Size", 3, 21, 5, step=2)
            z_threshold = st.slider("Z-Score Threshold", 1.0, 5.0, 3.0, step=0.1)
            use_median = st.checkbox("Use Median Filter", value=True)
            spikes_only = st.checkbox("Detect Spikes Only", value=False)
            
            classical_params = {
                "window_size": window_size,
                "z_threshold": z_threshold,
                "use_median": use_median,
                "spikes_only": spikes_only
            }
            
        elif cleaning_method == "Deep Learning Based":
            st.subheader("Deep Learning Parameters")
            
            epochs = st.slider("Training Epochs", 10, 100, 30)
            hidden_dim = st.slider("Hidden Dimension", 16, 128, 64, step=16)
            
            deep_params = {
                "epochs": epochs,
                "hidden_dim": hidden_dim,
                "learning_rate": 0.001
            }
            
        elif cleaning_method == "Quantum-Inspired":
            st.subheader("Quantum Parameters")
            
            n_qubits = st.slider("Number of Qubits", 2, 8, 4)
            circuit_depth = st.slider("Circuit Depth", 2, 10, 5)
            
            quantum_params = {
                "n_qubits": n_qubits,
                "circuit_depth": circuit_depth,
                "shots": 1000
            }
            
        elif cleaning_method == "Hybrid (Auto-select)":
            st.info("The system will automatically select the best method based on data characteristics.")
            
            # Still allow setting some parameters
            st.subheader("Classical Parameters")
            classical_params = {"window_size": st.slider("Window Size", 3, 21, 5, step=2)}
            
            st.subheader("Deep Learning Parameters")
            deep_params = {"epochs": st.slider("Training Epochs", 10, 100, 30)}
            
            st.subheader("Quantum Parameters")
            quantum_params = {"n_qubits": st.slider("Number of Qubits", 2, 8, 4)}
        
        # Clean button
        if st.button("Clean Data"):
            if st.session_state.selected_column:
                with st.spinner("Cleaning data..."):
                    try:
                        # Apply the selected method
                        if cleaning_method == "Classical Statistical Methods":
                            result = classical_cleaner.clean(
                                st.session_state.uploaded_df,
                                st.session_state.selected_column,
                                classical_params
                            )
                            method = "classical"
                        
                        elif cleaning_method == "Deep Learning Based":
                            result = deep_cleaner.clean(
                                st.session_state.uploaded_df,
                                st.session_state.selected_column,
                                deep_params
                            )
                            method = "deep_learning"
                        
                        elif cleaning_method == "Quantum-Inspired":
                            result = quantum_cleaner.clean(
                                st.session_state.uploaded_df,
                                st.session_state.selected_column,
                                quantum_params
                            )
                            method = "quantum"
                        
                        elif cleaning_method == "Hybrid (Auto-select)":
                            # First determine the best method
                            method, params = bandit_selector.select_best_method(
                                st.session_state.uploaded_df,
                                st.session_state.selected_column
                            )
                            
                            # Apply the selected method
                            if method == "classical":
                                result = classical_cleaner.clean(
                                    st.session_state.uploaded_df,
                                    st.session_state.selected_column,
                                    params
                                )
                            elif method == "deep_learning":
                                result = deep_cleaner.clean(
                                    st.session_state.uploaded_df,
                                    st.session_state.selected_column,
                                    params
                                )
                            elif method == "quantum":
                                result = quantum_cleaner.clean(
                                    st.session_state.uploaded_df,
                                    st.session_state.selected_column,
                                    params
                                )
                            
                            # In hybrid mode, update the reward
                            bandit_selector.update_reward(
                                method=method,
                                context=bandit_selector.extract_features(
                                    st.session_state.uploaded_df,
                                    st.session_state.selected_column
                                ),
                                reward=result.metadata.get("quality_score", 0.5)
                            )
                        
                        # Store the result
                        st.session_state.cleaned_df = result.df
                        
                        # Store metadata
                        metadata = result.metadata
                        metadata["method"] = method
                        metadata["selected_method"] = method if cleaning_method == "Hybrid (Auto-select)" else None
                        metadata["cleaning_time"] = metadata.get("execution_time_ms", 0)
                        st.session_state.metadata = metadata
                        
                        # Add log entry
                        add_log("info", f"Data cleaned successfully using {method} method in {metadata.get('execution_time_ms', 0):.1f} ms")
                        
                        # Add info about the cleaning
                        if "outliers_detected" in metadata:
                            add_log("info", f"Detected {metadata['outliers_detected']} outliers")
                        if "missing_filled" in metadata:
                            add_log("info", f"Filled {metadata['missing_filled']} missing values")
                            
                        st.success(f"Data cleaned successfully using {method.replace('_', ' ')} method!")
                    except Exception as e:
                        add_log("error", f"Error cleaning data: {str(e)}")
                        st.error(f"Error cleaning data: {str(e)}")
            else:
                st.error("Please select a column to clean.")

# Main content area - only show if data is loaded
if st.session_state.uploaded_df is not None:
    # Tabs for different views
    tabs = st.tabs(["Data Preview", "Cleaning Results", "Detailed Analysis", "Logs and Reports", "Database & Search", "OCR Extraction"])
    
    # Data Preview Tab
    with tabs[0]:
        st.subheader("Data Preview")
        st.dataframe(st.session_state.uploaded_df.head(20))
        
        # Basic statistics
        st.subheader("Data Statistics")
        if st.session_state.selected_column:
            stats = get_column_stats(st.session_state.uploaded_df, st.session_state.selected_column)
            
            # Create columns for stats display
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Mean", f"{stats['mean']:.2f}")
                st.metric("Median", f"{stats['median']:.2f}")
            
            with col2:
                st.metric("Min", f"{stats['min']:.2f}")
                st.metric("Max", f"{stats['max']:.2f}")
            
            with col3:
                st.metric("Std Dev", f"{stats['std']:.2f}")
                st.metric("Missing Values", f"{stats['missing']} ({stats['missing_pct']:.1f}%)")
        
        # Preview plot
        st.subheader("Time Series Plot")
        if st.session_state.selected_column:
            fig = plot_time_series(
                st.session_state.uploaded_df, 
                st.session_state.selected_column, 
                "Original Time Series Data"
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Select a column to visualize the time series.")
    
    # Cleaning Results Tab
    with tabs[1]:
        st.subheader("Cleaning Results")
        
        # Check if cleaned data is available
        if st.session_state.cleaned_df is not None:
            # Show original vs cleaned data
            st.markdown("### Original vs Cleaned Data")
            
            # Show statistics side by side
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Original Data Statistics**")
                original_stats = get_column_stats(
                    st.session_state.uploaded_df, 
                    st.session_state.selected_column
                )
                
                st.metric("Mean", f"{original_stats['mean']:.2f}")
                st.metric("Median", f"{original_stats['median']:.2f}")
                st.metric("Std Dev", f"{original_stats['std']:.2f}")
                st.metric("Missing Values", f"{original_stats['missing']} ({original_stats['missing_pct']:.1f}%)")
            
            with col2:
                st.markdown("**Cleaned Data Statistics**")
                cleaned_stats = get_column_stats(
                    st.session_state.cleaned_df, 
                    st.session_state.selected_column
                )
                
                st.metric("Mean", f"{cleaned_stats['mean']:.2f}")
                st.metric("Median", f"{cleaned_stats['median']:.2f}")
                st.metric("Std Dev", f"{cleaned_stats['std']:.2f}")
                st.metric("Missing Values", f"{cleaned_stats['missing']} ({cleaned_stats['missing_pct']:.1f}%)")
            
            # Plot original vs cleaned
            st.markdown("### Comparison Plot")
            
            # Create a figure with two subplots
            fig = go.Figure()
            
            # Add original data
            fig.add_trace(go.Scatter(
                y=st.session_state.uploaded_df[st.session_state.selected_column],
                mode='lines',
                name='Original Data',
                line=dict(color='blue', width=1)
            ))
            
            # Add cleaned data
            fig.add_trace(go.Scatter(
                y=st.session_state.cleaned_df[st.session_state.selected_column],
                mode='lines',
                name='Cleaned Data',
                line=dict(color='red', width=1)
            ))
            
            # Update layout
            fig.update_layout(
                title="Original vs Cleaned Data",
                height=500,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Display cleaning metadata if available
            if st.session_state.metadata:
                st.markdown("### Cleaning Details")
                
                # Create an expander for detailed information
                with st.expander("View cleaning details", expanded=True):
                    # Method information
                    method = st.session_state.metadata.get("method", "unknown")
                    if method == "hybrid":
                        selected_method = st.session_state.metadata.get("selected_method", "unknown")
                        st.info(f"Hybrid cleaning used {selected_method} method based on data characteristics")
                    
                    # Create columns for metadata display
                    meta_col1, meta_col2 = st.columns(2)
                    
                    with meta_col1:
                        st.markdown("**Process Information**")
                        st.write(f"Method: {method.replace('_', ' ').title()}")
                        st.write(f"Execution time: {st.session_state.metadata.get('execution_time_ms', 0):.1f} ms")
                        st.write(f"Job ID: {st.session_state.job_id}")
                    
                    with meta_col2:
                        st.markdown("**Results Summary**")
                        st.write(f"Outliers detected: {st.session_state.metadata.get('outliers_detected', 0)}")
                        st.write(f"Missing values filled: {st.session_state.metadata.get('missing_filled', 0)}")
                        st.write(f"Quality score: {st.session_state.metadata.get('quality_score', 0):.2f}")
                    
                    # Display additional method-specific parameters
                    st.markdown("**Method Parameters**")
                    params = {k: v for k, v in st.session_state.metadata.items() 
                            if k not in ['method', 'execution_time_ms', 'outliers_detected', 
                                        'missing_filled', 'quality_score', 'selected_method']}
                    st.json(params)
            
            # Download cleaned data
            st.markdown("### Download Cleaned Data")
            
            # Create a buffer and save the dataframe to it
            csv_buffer = io.StringIO()
            st.session_state.cleaned_df.to_csv(csv_buffer, index=False)
            
            # Add download button
            st.download_button(
                label="Download Cleaned Data as CSV",
                data=csv_buffer.getvalue(),
                file_name=f"cleaned_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
        else:
            st.info("Clean your data first to see the results.")
    
    # Detailed Analysis Tab
    with tabs[2]:
        st.subheader("Detailed Analysis")
        
        # Check if cleaned data is available
        if st.session_state.cleaned_df is not None:
            # Create three sections: Anomaly Analysis, Pattern Detection, and Distribution Analysis
            analysis_tabs = st.tabs(["Anomaly Analysis", "Pattern Detection", "Distribution Analysis"])
            
            # Anomaly Analysis
            with analysis_tabs[0]:
                st.markdown("### Anomaly Analysis")
                
                # Compare original vs cleaned anomalies
                anomaly_col1, anomaly_col2 = st.columns(2)
                
                with anomaly_col1:
                    st.markdown("**Original Data Anomalies**")
                    
                    # Run anomaly detection on original data
                    original_report = generate_quality_report(
                        st.session_state.uploaded_df, 
                        st.session_state.selected_column
                    )
                    
                    # Display results
                    st.metric("Z-Score Outliers", original_report["outliers_z_score"])
                    st.metric("IQR Outliers", original_report["outliers_iqr"])
                    st.metric("Potential Anomalies", original_report["potential_anomalies"])
                
                with anomaly_col2:
                    st.markdown("**Cleaned Data Anomalies**")
                    
                    # Run anomaly detection on cleaned data
                    cleaned_report = generate_quality_report(
                        st.session_state.cleaned_df, 
                        st.session_state.selected_column
                    )
                    
                    # Display results
                    st.metric("Z-Score Outliers", cleaned_report["outliers_z_score"])
                    st.metric("IQR Outliers", cleaned_report["outliers_iqr"])
                    st.metric("Potential Anomalies", cleaned_report["potential_anomalies"])
                
                # Highlight differences
                if original_report["outliers_z_score"] > cleaned_report["outliers_z_score"]:
                    st.success(f"Z-Score outliers reduced by {original_report['outliers_z_score'] - cleaned_report['outliers_z_score']}")
                
                if original_report["outliers_iqr"] > cleaned_report["outliers_iqr"]:
                    st.success(f"IQR outliers reduced by {original_report['outliers_iqr'] - cleaned_report['outliers_iqr']}")
                
                if original_report["potential_anomalies"] > cleaned_report["potential_anomalies"]:
                    st.success(f"Potential anomalies reduced by {original_report['potential_anomalies'] - cleaned_report['potential_anomalies']}")
            
            # Pattern Detection
            with analysis_tabs[1]:
                st.markdown("### Pattern Detection")
                
                # Generate autocorrelation plot
                st.markdown("#### Autocorrelation Analysis")
                
                # Create figure with two subplots
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
                
                # Original data autocorrelation
                original_series = st.session_state.uploaded_df[st.session_state.selected_column].dropna()
                if len(original_series) > 5:  # Need at least 5 points for a meaningful autocorrelation
                    pd.plotting.autocorrelation_plot(original_series, ax=ax1)
                    ax1.set_title("Original Data Autocorrelation")
                else:
                    ax1.text(0.5, 0.5, "Not enough data for autocorrelation", ha='center', va='center')
                
                # Cleaned data autocorrelation
                cleaned_series = st.session_state.cleaned_df[st.session_state.selected_column].dropna()
                if len(cleaned_series) > 5:  # Need at least 5 points for a meaningful autocorrelation
                    pd.plotting.autocorrelation_plot(cleaned_series, ax=ax2)
                    ax2.set_title("Cleaned Data Autocorrelation")
                else:
                    ax2.text(0.5, 0.5, "Not enough data for autocorrelation", ha='center', va='center')
                
                # Adjust layout
                plt.tight_layout()
                
                # Display plot
                st.pyplot(fig)
                
                # Check for seasonality
                st.markdown("#### Seasonality Detection")
                
                # Detect seasonality in original and cleaned data
                try:
                    # We'll use a simple method to detect seasonality:
                    # Find peaks in autocorrelation function
                    from scipy import signal
                    
                    # Original data
                    if len(original_series) > 10:
                        acf_orig = pd.plotting.autocorrelation_plot(original_series, ax=plt.gca())
                        plt.close()
                        
                        # Get autocorrelation values from the returned Line2D object
                        acf_values = acf_orig.get_ydata()
                        
                        # Find peaks
                        peaks, _ = signal.find_peaks(acf_values)
                        if len(peaks) > 1:
                            # Convert lags to periods
                            periods = peaks[1:] - peaks[:-1]
                            avg_period = sum(periods) / len(periods)
                            st.info(f"Detected possible seasonality with period of approximately {avg_period:.1f} time units in original data")
                        else:
                            st.info("No clear seasonality detected in original data")
                    else:
                        st.info("Not enough data points to detect seasonality in original data")
                    
                    # Cleaned data
                    if len(cleaned_series) > 10:
                        acf_clean = pd.plotting.autocorrelation_plot(cleaned_series, ax=plt.gca())
                        plt.close()
                        
                        # Get autocorrelation values
                        acf_values = acf_clean.get_ydata()
                        
                        # Find peaks
                        peaks, _ = signal.find_peaks(acf_values)
                        if len(peaks) > 1:
                            # Convert lags to periods
                            periods = peaks[1:] - peaks[:-1]
                            avg_period = sum(periods) / len(periods)
                            st.info(f"Detected possible seasonality with period of approximately {avg_period:.1f} time units in cleaned data")
                        else:
                            st.info("No clear seasonality detected in cleaned data")
                    else:
                        st.info("Not enough data points to detect seasonality in cleaned data")
                    
                except Exception as e:
                    st.warning(f"Could not perform seasonality detection: {str(e)}")
            
            # Distribution Analysis
            with analysis_tabs[2]:
                st.markdown("### Distribution Analysis")
                
                # Create histograms for original and cleaned data
                fig = go.Figure()
                
                # Add histogram for original data
                fig.add_trace(go.Histogram(
                    x=st.session_state.uploaded_df[st.session_state.selected_column].dropna(),
                    name="Original Data",
                    opacity=0.75,
                    marker_color='blue'
                ))
                
                # Add histogram for cleaned data
                fig.add_trace(go.Histogram(
                    x=st.session_state.cleaned_df[st.session_state.selected_column].dropna(),
                    name="Cleaned Data",
                    opacity=0.75,
                    marker_color='red'
                ))
                
                # Update layout
                fig.update_layout(
                    title="Distribution Comparison",
                    xaxis_title=st.session_state.selected_column,
                    yaxis_title="Count",
                    barmode='overlay',
                    height=500
                )
                
                # Display plot
                st.plotly_chart(fig, use_container_width=True)
                
                # Normality test
                st.markdown("#### Normality Test")
                
                try:
                    from scipy import stats
                    
                    # Test original data
                    original_data = st.session_state.uploaded_df[st.session_state.selected_column].dropna()
                    if len(original_data) > 8:  # Minimum sample size for meaningful test
                        stat, p_value = stats.shapiro(original_data)
                        if p_value > 0.05:
                            st.info(f"Original data appears to be normally distributed (Shapiro-Wilk p-value: {p_value:.4f})")
                        else:
                            st.info(f"Original data does not appear to be normally distributed (Shapiro-Wilk p-value: {p_value:.4f})")
                    else:
                        st.info("Not enough data points for normality test on original data")
                    
                    # Test cleaned data
                    cleaned_data = st.session_state.cleaned_df[st.session_state.selected_column].dropna()
                    if len(cleaned_data) > 8:  # Minimum sample size for meaningful test
                        stat, p_value = stats.shapiro(cleaned_data)
                        if p_value > 0.05:
                            st.info(f"Cleaned data appears to be normally distributed (Shapiro-Wilk p-value: {p_value:.4f})")
                        else:
                            st.info(f"Cleaned data does not appear to be normally distributed (Shapiro-Wilk p-value: {p_value:.4f})")
                    else:
                        st.info("Not enough data points for normality test on cleaned data")
                
                except Exception as e:
                    st.warning(f"Could not perform normality test: {str(e)}")
                
                # Basic statistics comparison
                st.markdown("#### Statistical Comparison")
                
                # Create a table with statistics
                stats_df = pd.DataFrame({
                    "Statistic": ["Mean", "Median", "Std Dev", "Min", "Max", "Skewness", "Kurtosis"],
                    "Original": [
                        f"{original_series.mean():.4f}",
                        f"{original_series.median():.4f}",
                        f"{original_series.std():.4f}",
                        f"{original_series.min():.4f}",
                        f"{original_series.max():.4f}",
                        f"{original_series.skew():.4f}",
                        f"{original_series.kurtosis():.4f}"
                    ],
                    "Cleaned": [
                        f"{cleaned_series.mean():.4f}",
                        f"{cleaned_series.median():.4f}",
                        f"{cleaned_series.std():.4f}",
                        f"{cleaned_series.min():.4f}",
                        f"{cleaned_series.max():.4f}",
                        f"{cleaned_series.skew():.4f}",
                        f"{cleaned_series.kurtosis():.4f}"
                    ]
                })
                
                # Display table
                st.table(stats_df)
        else:
            st.info("Clean your data first to see detailed analysis.")
    
    # Logs and Reports Tab
    with tabs[3]:
        st.subheader("Logs and Reports")
        
        # Create tabs for logs and reports
        log_tabs = st.tabs(["Activity Logs", "Quality Report", "Performance Metrics"])
        
        # Activity Logs
        with log_tabs[0]:
            st.markdown("### Activity Logs")
            
            # Check if there are logs
            if st.session_state.logs:
                # Create a dataframe from logs
                logs_df = pd.DataFrame(st.session_state.logs)
                
                # Add filter options
                log_level = st.multiselect(
                    "Filter by log level",
                    options=sorted(logs_df["level"].unique()),
                    default=sorted(logs_df["level"].unique())
                )
                
                # Apply filter
                filtered_logs = logs_df[logs_df["level"].isin(log_level)]
                
                # Display logs
                for _, log in filtered_logs.iterrows():
                    if log["level"] == "error":
                        st.error(f"{log['timestamp']} - {log['message']}")
                    elif log["level"] == "warning":
                        st.warning(f"{log['timestamp']} - {log['message']}")
                    elif log["level"] == "info":
                        st.info(f"{log['timestamp']} - {log['message']}")
                    else:
                        st.text(f"{log['timestamp']} - {log['level']} - {log['message']}")
                
                # Add download button for logs
                csv_buffer = io.StringIO()
                logs_df.to_csv(csv_buffer, index=False)
                
                st.download_button(
                    label="Download Logs as CSV",
                    data=csv_buffer.getvalue(),
                    file_name=f"cleaning_logs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
            else:
                st.info("No logs available yet.")
        
        # Quality Report
        with log_tabs[1]:
            st.markdown("### Data Quality Report")
            
            # Check if cleaned data is available
            if st.session_state.cleaned_df is not None:
                # Generate quality report
                st.markdown("#### Data Quality Metrics")
                
                # Create a comprehensive report
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Original Data**")
                    
                    # Generate report
                    original_report = generate_quality_report(
                        st.session_state.uploaded_df, 
                        st.session_state.selected_column
                    )
                    
                    # Display metrics
                    st.metric("Missing Values", f"{original_report['missing_values']} ({original_report['missing_percentage']:.1f}%)")
                    st.metric("Outliers (Z-Score)", original_report["outliers_z_score"])
                    st.metric("Outliers (IQR)", original_report["outliers_iqr"])
                    st.metric("Potential Anomalies", original_report["potential_anomalies"])
                
                with col2:
                    st.markdown("**Cleaned Data**")
                    
                    # Generate report
                    cleaned_report = generate_quality_report(
                        st.session_state.cleaned_df, 
                        st.session_state.selected_column
                    )
                    
                    # Display metrics
                    st.metric("Missing Values", f"{cleaned_report['missing_values']} ({cleaned_report['missing_percentage']:.1f}%)")
                    st.metric("Outliers (Z-Score)", cleaned_report["outliers_z_score"])
                    st.metric("Outliers (IQR)", cleaned_report["outliers_iqr"])
                    st.metric("Potential Anomalies", cleaned_report["potential_anomalies"])
                
                # Quality score
                quality_score = 1.0
                
                # Decrease score for each remaining issue
                if cleaned_report["missing_values"] > 0:
                    quality_score -= min(0.3, cleaned_report["missing_percentage"] / 100)
                
                if cleaned_report["outliers_z_score"] > 0:
                    quality_score -= min(0.3, cleaned_report["outliers_z_score"] / max(1, original_report["outliers_z_score"]) * 0.3)
                
                if cleaned_report["potential_anomalies"] > 0:
                    quality_score -= min(0.2, cleaned_report["potential_anomalies"] / max(1, original_report["potential_anomalies"]) * 0.2)
                
                # Ensure score is between 0 and 1
                quality_score = max(0, min(1, quality_score))
                
                # Display quality score
                st.markdown("#### Overall Quality Score")
                
                # Create a gauge chart
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=quality_score * 100,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "Data Quality Score"},
                    gauge={
                        'axis': {'range': [0, 100]},
                        'bar': {'color': "darkblue"},
                        'steps': [
                            {'range': [0, 50], 'color': "red"},
                            {'range': [50, 75], 'color': "orange"},
                            {'range': [75, 100], 'color': "green"}
                        ]
                    }
                ))
                
                fig.update_layout(height=300)
                
                # Display gauge
                st.plotly_chart(fig, use_container_width=True)
                
                # Interpretation
                if quality_score >= 0.9:
                    st.success("Your data is of excellent quality with very few issues.")
                elif quality_score >= 0.75:
                    st.success("Your data is of good quality with minor issues.")
                elif quality_score >= 0.5:
                    st.warning("Your data has moderate quality issues that may need further attention.")
                else:
                    st.error("Your data has significant quality issues that require attention.")
                
                # Generate PDF report button
                if st.button("Generate PDF Report"):
                    try:
                        from fpdf import FPDF
                        
                        # Create PDF
                        pdf = FPDF()
                        pdf.add_page()
                        
                        # Add title
                        pdf.set_font("Arial", "B", 16)
                        pdf.cell(0, 10, "Time Series Data Quality Report", ln=True, align="C")
                        pdf.ln(10)
                        
                        # Add date
                        pdf.set_font("Arial", "", 10)
                        pdf.cell(0, 10, f"Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True)
                        pdf.ln(5)
                        
                        # Add job ID
                        pdf.cell(0, 10, f"Job ID: {st.session_state.job_id}", ln=True)
                        pdf.ln(10)
                        
                        # Add quality score
                        pdf.set_font("Arial", "B", 14)
                        pdf.cell(0, 10, f"Overall Quality Score: {quality_score * 100:.1f}%", ln=True)
                        pdf.ln(10)
                        
                        # Add comparative metrics
                        pdf.set_font("Arial", "B", 12)
                        pdf.cell(0, 10, "Data Quality Metrics", ln=True)
                        pdf.ln(5)
                        
                        # Add table header
                        pdf.set_font("Arial", "B", 10)
                        pdf.cell(60, 10, "Metric", border=1)
                        pdf.cell(40, 10, "Original", border=1)
                        pdf.cell(40, 10, "Cleaned", border=1)
                        pdf.cell(40, 10, "Improvement", border=1)
                        pdf.ln()
                        
                        # Add table rows
                        pdf.set_font("Arial", "", 10)
                        
                        # Missing values
                        pdf.cell(60, 10, "Missing Values", border=1)
                        pdf.cell(40, 10, f"{original_report['missing_values']} ({original_report['missing_percentage']:.1f}%)", border=1)
                        pdf.cell(40, 10, f"{cleaned_report['missing_values']} ({cleaned_report['missing_percentage']:.1f}%)", border=1)
                        missing_improvement = original_report['missing_values'] - cleaned_report['missing_values']
                        pdf.cell(40, 10, f"{missing_improvement} ({missing_improvement / max(1, original_report['missing_values']) * 100:.1f}%)", border=1)
                        pdf.ln()
                        
                        # Outliers Z-Score
                        pdf.cell(60, 10, "Outliers (Z-Score)", border=1)
                        pdf.cell(40, 10, f"{original_report['outliers_z_score']}", border=1)
                        pdf.cell(40, 10, f"{cleaned_report['outliers_z_score']}", border=1)
                        z_improvement = original_report['outliers_z_score'] - cleaned_report['outliers_z_score']
                        pdf.cell(40, 10, f"{z_improvement} ({z_improvement / max(1, original_report['outliers_z_score']) * 100:.1f}%)", border=1)
                        pdf.ln()
                        
                        # Outliers IQR
                        pdf.cell(60, 10, "Outliers (IQR)", border=1)
                        pdf.cell(40, 10, f"{original_report['outliers_iqr']}", border=1)
                        pdf.cell(40, 10, f"{cleaned_report['outliers_iqr']}", border=1)
                        iqr_improvement = original_report['outliers_iqr'] - cleaned_report['outliers_iqr']
                        pdf.cell(40, 10, f"{iqr_improvement} ({iqr_improvement / max(1, original_report['outliers_iqr']) * 100:.1f}%)", border=1)
                        pdf.ln()
                        
                        # Potential Anomalies
                        pdf.cell(60, 10, "Potential Anomalies", border=1)
                        pdf.cell(40, 10, f"{original_report['potential_anomalies']}", border=1)
                        pdf.cell(40, 10, f"{cleaned_report['potential_anomalies']}", border=1)
                        anom_improvement = original_report['potential_anomalies'] - cleaned_report['potential_anomalies']
                        pdf.cell(40, 10, f"{anom_improvement} ({anom_improvement / max(1, original_report['potential_anomalies']) * 100:.1f}%)", border=1)
                        pdf.ln(20)
                        
                        # Add method information
                        pdf.set_font("Arial", "B", 12)
                        pdf.cell(0, 10, "Cleaning Method Information", ln=True)
                        pdf.ln(5)
                        
                        # Method details
                        pdf.set_font("Arial", "", 10)
                        method = st.session_state.metadata.get("method", "unknown")
                        pdf.cell(0, 10, f"Method: {method.replace('_', ' ').title()}", ln=True)
                        
                        if method == "hybrid" and "selected_method" in st.session_state.metadata:
                            pdf.cell(0, 10, f"Selected Sub-Method: {st.session_state.metadata['selected_method'].replace('_', ' ').title()}", ln=True)
                        
                        pdf.cell(0, 10, f"Execution Time: {st.session_state.metadata.get('execution_time_ms', 0):.1f} ms", ln=True)
                        pdf.cell(0, 10, f"Outliers Detected: {st.session_state.metadata.get('outliers_detected', 0)}", ln=True)
                        pdf.cell(0, 10, f"Missing Values Filled: {st.session_state.metadata.get('missing_filled', 0)}", ln=True)
                        
                        # Save PDF to a temp file
                        temp_pdf = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
                        pdf_path = temp_pdf.name
                        pdf.output(pdf_path)
                        
                        # Read the PDF file
                        with open(pdf_path, "rb") as f:
                            pdf_bytes = f.read()
                        
                        # Create download button
                        st.download_button(
                            label="Download PDF Report",
                            data=pdf_bytes,
                            file_name=f"quality_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                            mime="application/pdf"
                        )
                        
                        # Clean up
                        os.unlink(pdf_path)
                        
                        st.success("PDF report generated successfully!")
                    except Exception as e:
                        st.error(f"Error generating PDF report: {str(e)}")
            else:
                st.info("Clean your data first to generate a quality report.")
        
        # Performance Metrics
        with log_tabs[2]:
            st.markdown("### Performance Metrics")
            
            # Check if cleaned data is available
            if st.session_state.cleaned_df is not None and st.session_state.metadata:
                # Display performance metrics
                st.markdown("#### Cleaning Performance")
                
                # Get execution time
                execution_time = st.session_state.metadata.get("execution_time_ms", 0)
                
                # Create gauge chart for execution time
                fig = go.Figure(go.Indicator(
                    mode="gauge+number+delta",
                    value=execution_time,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "Execution Time (ms)"},
                    delta={'reference': 1000, 'increasing': {'color': "red"}, 'decreasing': {'color': "green"}},
                    gauge={
                        'axis': {'range': [0, 2000]},
                        'bar': {'color': "darkblue"},
                        'steps': [
                            {'range': [0, 500], 'color': "green"},
                            {'range': [500, 1000], 'color': "yellow"},
                            {'range': [1000, 2000], 'color': "red"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 1000
                        }
                    }
                ))
                
                fig.update_layout(height=300)
                
                # Display gauge
                st.plotly_chart(fig, use_container_width=True)
                
                # Create metrics comparison for all cleaning methods
                st.markdown("#### Method Comparison")
                
                # Create data for comparison
                methods = ["classical", "deep_learning", "quantum", "hybrid"]
                execution_times = [st.session_state.metadata.get("execution_time_ms", 0) if st.session_state.metadata.get("method") == method else 0 for method in methods]
                
                # Create bar chart
                fig = px.bar(
                    x=methods,
                    y=execution_times,
                    labels={"x": "Method", "y": "Execution Time (ms)"},
                    title="Method Execution Time Comparison"
                )
                
                # Add annotation for the used method
                fig.add_annotation(
                    x=st.session_state.metadata.get("method", "unknown"),
                    y=st.session_state.metadata.get("execution_time_ms", 0),
                    text="Used Method",
                    showarrow=True,
                    arrowhead=1
                )
                
                # Update layout
                fig.update_layout(height=400)
                
                # Display bar chart
                st.plotly_chart(fig, use_container_width=True)
                
                # Display method-specific metrics
                method = st.session_state.metadata.get("method", "unknown")
                
                if method == "classical":
                    st.markdown("#### Classical Method Metrics")
                    if "window_size" in st.session_state.metadata:
                        st.metric("Window Size", st.session_state.metadata["window_size"])
                    if "z_threshold" in st.session_state.metadata:
                        st.metric("Z-Score Threshold", st.session_state.metadata["z_threshold"])
                
                elif method == "deep_learning":
                    st.markdown("#### Deep Learning Method Metrics")
                    if "epochs" in st.session_state.metadata:
                        st.metric("Training Epochs", st.session_state.metadata["epochs"])
                    if "loss" in st.session_state.metadata:
                        st.metric("Final Loss", f"{st.session_state.metadata['loss']:.4f}")
                
                elif method == "quantum":
                    st.markdown("#### Quantum Method Metrics")
                    if "n_qubits" in st.session_state.metadata:
                        st.metric("Number of Qubits", st.session_state.metadata["n_qubits"])
                    if "circuit_depth" in st.session_state.metadata:
                        st.metric("Circuit Depth", st.session_state.metadata["circuit_depth"])
                    if "shots" in st.session_state.metadata:
                        st.metric("Shots", st.session_state.metadata["shots"])
                
                elif method == "hybrid":
                    st.markdown("#### Hybrid Method Metrics")
                    if "selected_method" in st.session_state.metadata:
                        st.metric("Selected Method", st.session_state.metadata["selected_method"])
                    if "selection_time_ms" in st.session_state.metadata:
                        st.metric("Method Selection Time", f"{st.session_state.metadata['selection_time_ms']:.1f} ms")
            else:
                st.info("Clean your data first to see performance metrics.")
    
    # Database & Search Tab
    with tabs[4]:
        st.subheader("Database & Vector Search")
        
        # Create tabs for database operations
        tab1, tab2 = st.tabs(["Store & Manage Data", "Search & Retrieve"])
        
        # Store & Manage tab
        with tab1:
            st.markdown("### Store Time Series in Database")
            
            # Check if cleaned data is available
            if st.session_state.cleaned_df is not None:
                # Form for storing time series
                with st.form("store_form"):
                    description = st.text_area("Description", 
                                             "Time series data cleaned with Advanced Hybrid Time-Series Cleaning System", 
                                             help="Provide a description for this time series data")
                    
                    tags = st.text_input("Tags (comma-separated)", 
                                        "cleaned, time-series", 
                                        help="Add tags to help with later retrieval")
                    
                    source = st.text_input("Data Source", 
                                          "Custom upload", 
                                          help="Where did this data come from?")
                    
                    # Store metadata about the cleaning method
                    method_info = "unknown"
                    if "method" in st.session_state.metadata:
                        method_info = st.session_state.metadata["method"]
                        if method_info == "hybrid" and "selected_method" in st.session_state.metadata:
                            method_info += f"-{st.session_state.metadata['selected_method']}"
                    
                    # Submit button
                    submit = st.form_submit_button("Store in Database")
                    
                    if submit:
                        try:
                            # Prepare metadata
                            metadata = {
                                "description": description,
                                "tags": tags,
                                "source": source,
                                "cleaning_method": method_info,
                                "stored_date": datetime.now().isoformat(),
                                "user_comment": "Stored via Hybrid Time-Series Cleaning System"
                            }
                            
                            # Add cleaning metadata if available
                            if st.session_state.metadata:
                                for key, value in st.session_state.metadata.items():
                                    if isinstance(value, (str, int, float, bool)) and key not in metadata:
                                        metadata[key] = value
                            
                            # Store in database
                            job_id = db_service.store_time_series(
                                df=st.session_state.cleaned_df,
                                column=st.session_state.selected_column,
                                metadata=metadata,
                                job_id=st.session_state.job_id
                            )
                            
                            st.success(f"Successfully stored time series in database with ID: {job_id}")
                        except Exception as e:
                            st.error(f"Error storing time series: {str(e)}")
                
                # Display divider
                st.markdown("---")
            else:
                st.info("Clean your data first to store it in the database.")
            
            # List and manage stored time series
            st.markdown("### Manage Stored Data")
            
            if st.button("Refresh Database Entries"):
                stored_items = db_service.list_all_time_series()
                if stored_items:
                    st.session_state.stored_items = stored_items
                    st.success(f"Found {len(stored_items)} stored time series.")
                else:
                    st.info("No time series found in the database.")
                    st.session_state.stored_items = []
            
            # Display stored items if available
            if "stored_items" in st.session_state and st.session_state.stored_items:
                st.markdown("#### Stored Time Series")
                
                for i, item in enumerate(st.session_state.stored_items):
                    with st.expander(f"{i+1}. {item['job_id']} - {item['metadata'].get('description', 'No description')}"):
                        st.markdown(f"**Description:** {item['description']}")
                        
                        # Create two columns for metadata display
                        col1, col2 = st.columns(2)
                        
                        # Display metadata
                        with col1:
                            st.markdown("**Metadata:**")
                            meta_df = pd.DataFrame(
                                [(k, str(v)) for k, v in item["metadata"].items() if k not in ["description"]],
                                columns=["Property", "Value"]
                            )
                            st.dataframe(meta_df, use_container_width=True)
                        
                        # Add delete button
                        with col2:
                            if st.button(f"Delete Entry", key=f"delete_{item['job_id']}"):
                                if db_service.delete_time_series(item['job_id']):
                                    st.success(f"Successfully deleted time series {item['job_id']}")
                                    # Remove from session state list
                                    st.session_state.stored_items = [i for i in st.session_state.stored_items 
                                                                if i['job_id'] != item['job_id']]
                                    st.rerun()
                                else:
                                    st.error(f"Failed to delete time series {item['job_id']}")
            else:
                st.info("Click 'Refresh Database Entries' to view stored time series data.")
                
        # Search & Retrieve tab
        with tab2:
            st.markdown("### Find Similar Time Series")
            
            # Two search options
            search_method = st.radio("Search Method", 
                                    ["Find Similar to Current Data", "Search by Metadata"])
            
            if search_method == "Find Similar to Current Data":
                if st.session_state.uploaded_df is not None:
                    n_results = st.slider("Number of Results", 1, 10, 3)
                    
                    if st.button("Find Similar Time Series"):
                        with st.spinner("Searching for similar time series..."):
                            try:
                                results = db_service.find_similar_time_series(
                                    df=st.session_state.uploaded_df,
                                    column=st.session_state.selected_column,
                                    n_results=n_results
                                )
                                
                                if results:
                                    st.success(f"Found {len(results)} similar time series.")
                                    st.session_state.similar_results = results
                                else:
                                    st.info("No similar time series found.")
                                    st.session_state.similar_results = []
                            except Exception as e:
                                st.error(f"Error searching for similar time series: {str(e)}")
                    
                    # Display results if available
                    if "similar_results" in st.session_state and st.session_state.similar_results:
                        st.markdown("#### Similar Time Series Results")
                        
                        for i, result in enumerate(st.session_state.similar_results):
                            similarity = result.get("similarity", 0) * 100
                            with st.expander(f"{i+1}. Similarity: {similarity:.1f}% - {result['job_id']}"):
                                st.markdown(f"**Description:** {result['description']}")
                                
                                # Show metadata
                                st.markdown("**Metadata:**")
                                meta_df = pd.DataFrame(
                                    [(k, str(v)) for k, v in result["metadata"].items() if k not in ["description"]],
                                    columns=["Property", "Value"]
                                )
                                st.dataframe(meta_df, use_container_width=True)
                else:
                    st.info("Upload or load example data first to search for similar time series.")
            
            elif search_method == "Search by Metadata":
                st.markdown("### Search by Metadata")
                
                # Create a search form
                with st.form("metadata_search_form"):
                    # Common metadata search fields
                    cleaning_method = st.selectbox("Cleaning Method", 
                                                ["Any", "classical", "deep_learning", "quantum", "hybrid"])
                    
                    tags_input = st.text_input("Tags (comma-separated)", 
                                            "",
                                            help="Enter tags to search for")
                    
                    # Date range for stored_date if available
                    date_filter = st.checkbox("Filter by Storage Date")
                    
                    if date_filter:
                        today = datetime.now()
                        start_date = st.date_input("Start Date", today - timedelta(days=30))
                        end_date = st.date_input("End Date", today)
                    
                    # Submit button
                    submit_search = st.form_submit_button("Search")
                    
                    if submit_search:
                        # Prepare filter
                        metadata_filter = {}
                        
                        if cleaning_method != "Any":
                            metadata_filter["cleaning_method"] = cleaning_method
                        
                        if tags_input:
                            # We'll do a simple substring check in the application
                            # since ChromaDB doesn't support substring search well
                            metadata_filter["_tags"] = tags_input
                            
                        if date_filter:
                            # We'll handle date filtering in the application
                            # Store the date range in session state to filter results
                            st.session_state.date_range = {
                                "start": start_date,
                                "end": end_date
                            }
                        
                        # Search the database
                        try:
                            results = db_service.search_by_metadata(metadata_filter)
                            
                            # Post-process results
                            filtered_results = []
                            for result in results:
                                # Apply tag filtering if specified
                                if tags_input and "_tags" in metadata_filter:
                                    tags = result["metadata"].get("tags", "")
                                    if not any(tag.strip() in tags for tag in tags_input.split(",")):
                                        continue
                                
                                # Apply date filtering if specified
                                if date_filter and "date_range" in st.session_state:
                                    stored_date_str = result["metadata"].get("stored_date", "")
                                    if stored_date_str:
                                        try:
                                            stored_date = datetime.fromisoformat(stored_date_str).date()
                                            if stored_date < st.session_state.date_range["start"] or stored_date > st.session_state.date_range["end"]:
                                                continue
                                        except:
                                            # Ignore date filtering for this item if parsing fails
                                            pass
                                
                                # Add to filtered results
                                filtered_results.append(result)
                            
                            # Store and display results
                            if filtered_results:
                                st.success(f"Found {len(filtered_results)} matching time series.")
                                st.session_state.metadata_results = filtered_results
                            else:
                                st.info("No matching time series found.")
                                st.session_state.metadata_results = []
                        except Exception as e:
                            st.error(f"Error searching for time series: {str(e)}")
                
                # Display results if available
                if "metadata_results" in st.session_state and st.session_state.metadata_results:
                    st.markdown("#### Search Results")
                    
                    for i, result in enumerate(st.session_state.metadata_results):
                        with st.expander(f"{i+1}. {result['job_id']} - {result['metadata'].get('description', 'No description')}"):
                            # Show metadata
                            st.markdown("**Metadata:**")
                            meta_df = pd.DataFrame(
                                [(k, str(v)) for k, v in result["metadata"].items()],
                                columns=["Property", "Value"]
                            )
                            st.dataframe(meta_df, use_container_width=True)
                            
                            # Add button to load this time series
                            if st.button(f"Load This Time Series", key=f"load_{result['job_id']}"):
                                try:
                                    # Retrieve the time series
                                    ts_data = db_service.get_time_series(result['job_id'])
                                    
                                    if ts_data and "df" in ts_data and "column" in ts_data:
                                        # Store in session state
                                        st.session_state.uploaded_df = ts_data["df"]
                                        st.session_state.numeric_columns = ts_data["df"].select_dtypes(include=[np.number]).columns.tolist()
                                        st.session_state.selected_column = ts_data["column"]
                                        
                                        # Reset cleaned dataframe
                                        st.session_state.cleaned_df = None
                                        
                                        # Add log entry
                                        add_log("info", f"Loaded time series from database: {result['job_id']}")
                                        
                                        st.success(f"Successfully loaded time series {result['job_id']}.")
                                        st.rerun()
                                    else:
                                        st.error("Retrieved time series data is incomplete.")
                                except Exception as e:
                                    st.error(f"Error loading time series: {str(e)}")

    # OCR Extraction Tab
    with tabs[5]:
        st.subheader("OCR Time-Series Extraction")
        
        # Introduction
        st.markdown("""
        This tab shows the results of OCR extraction from images and PDFs. 
        Use the OCR Extract option in the sidebar to upload images or PDFs containing time-series data.
        """)
        
        # Check if OCR extraction was performed
        if "ocr_source_type" in st.session_state and st.session_state.ocr_source_type is not None:
            # Display extraction information
            st.markdown("### Extraction Information")
            
            # Display source type and quantum usage
            st.info(f"Source Type: {st.session_state.ocr_source_type.capitalize()}")
            if st.session_state.ocr_use_quantum:
                st.success("Used Quantum-Inspired Enhancement for better extraction quality")
            
            # Display metadata
            if st.session_state.ocr_metadata:
                st.markdown("### Extraction Metadata")
                
                metadata_expander = st.expander("View Extraction Details", expanded=True)
                with metadata_expander:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.metric("Execution Time", f"{st.session_state.ocr_metadata.get('execution_time_ms', 0):.1f} ms")
                        if "tables_found" in st.session_state.ocr_metadata:
                            st.metric("Tables Found", st.session_state.ocr_metadata["tables_found"])
                    
                    with col2:
                        if "date_column" in st.session_state.ocr_metadata and st.session_state.ocr_metadata["date_column"]:
                            st.metric("Date Column", st.session_state.ocr_metadata["date_column"])
                        if "value_column" in st.session_state.ocr_metadata and st.session_state.ocr_metadata["value_column"]:
                            st.metric("Value Column", st.session_state.ocr_metadata["value_column"])
                
                # Add success message if time series was found
                if st.session_state.ocr_metadata.get("time_series_found", False):
                    st.success("Time-series data successfully extracted!")
                else:
                    st.warning("No time-series data could be identified in the source.")
            
            # Display tabs for different views
            ocr_tabs = st.tabs(["Source Display", "Extracted Text", "Extracted Data"])
            
            # Source Display Tab
            with ocr_tabs[0]:
                if st.session_state.ocr_source_type == "image" and st.session_state.ocr_image is not None:
                    st.markdown("### Source Image")
                    st.image(st.session_state.ocr_image, caption="Source Image", use_column_width=True)
                elif st.session_state.ocr_source_type == "pdf":
                    st.markdown("### Source PDF")
                    st.info("PDF document was processed for OCR extraction.")
            
            # Extracted Text Tab
            with ocr_tabs[1]:
                if st.session_state.ocr_extracted_text:
                    st.markdown("### Extracted Text from OCR")
                    
                    # Format and display the text
                    st.code(st.session_state.ocr_extracted_text, language="text")
                    
                    # Add download button for extracted text
                    text_bytes = st.session_state.ocr_extracted_text.encode()
                    st.download_button(
                        label="Download Extracted Text",
                        data=text_bytes,
                        file_name=f"ocr_extracted_text_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                        mime="text/plain"
                    )
                else:
                    st.info("No text was extracted from the source.")
            
            # Extracted Data Tab
            with ocr_tabs[2]:
                if st.session_state.ocr_extracted_df is not None:
                    st.markdown("### Extracted Time-Series Data")
                    
                    # Show dataframe preview
                    st.dataframe(st.session_state.ocr_extracted_df.head(20))
                    
                    # Show a plot if data is available
                    if len(st.session_state.ocr_extracted_df) > 0:
                        st.markdown("### Data Visualization")
                        
                        # Select a numeric column for plotting
                        numeric_cols = st.session_state.ocr_extracted_df.select_dtypes(include=[np.number]).columns.tolist()
                        if numeric_cols:
                            value_col = st.session_state.ocr_metadata.get("value_column", numeric_cols[0])
                            if value_col not in numeric_cols:
                                value_col = numeric_cols[0]
                            
                            # Create plot
                            fig = px.line(
                                st.session_state.ocr_extracted_df,
                                y=value_col,
                                title=f"Extracted Time Series: {value_col}"
                            )
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Add download button for extracted data
                            csv_buffer = io.StringIO()
                            st.session_state.ocr_extracted_df.to_csv(csv_buffer)
                            st.download_button(
                                label="Download Extracted Data (CSV)",
                                data=csv_buffer.getvalue(),
                                file_name=f"ocr_extracted_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                mime="text/csv"
                            )
                        else:
                            st.warning("No numeric columns found for plotting.")
                    else:
                        st.warning("Extracted dataframe is empty.")
                else:
                    st.info("No structured data was extracted from the source.")
        else:
            st.info("No OCR extraction has been performed. Use the 'OCR Extract' option in the sidebar to upload images or PDFs.")

else:
    # Display guidance when no data is loaded
    st.info("Upload time-series data using the sidebar controls or select OCR extraction to get started.")
    
    # Show some explanations about the system
    st.markdown("""
    ### Available Features
    
    This advanced time-series cleaning system provides the following features:
    
    - **Classical Statistical Methods**: Moving average, median filtering, z-score based outlier detection, 
      and interpolation for missing values
    
    - **Deep Learning Based**: Neural network models to detect and correct anomalies
    
    - **Quantum-Inspired**: Quantum-inspired algorithms for anomaly detection and data cleaning
    
    - **Hybrid Approach**: Automatically selects the best method for your specific data
    
    - **OCR Extraction**: Extract time-series data from images and PDFs
    
    - **Database & Search**: Store cleaned time series and search for similar patterns
    
    ### Getting Started
    
    1. Upload your time-series data using the sidebar controls
    2. Select a column to clean
    3. Choose a cleaning method and configure parameters
    4. Click "Clean Data" to process your time series
    5. Explore results across different tabs
    6. Store and search time series in the database
    
    ### OCR Extraction
    
    To extract time-series data from images or PDFs:
    
    1. Select "OCR Extract" in the sidebar
    2. Choose the source type (Image or PDF)
    3. Upload your file
    4. Optionally enable quantum-inspired enhancement for better quality
    5. Click "Extract Data" to process the file
    6. View results in the OCR Extraction tab
    """)