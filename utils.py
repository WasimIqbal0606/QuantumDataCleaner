import pandas as pd
import numpy as np
import os
import time
import json
import base64
import logging
import matplotlib.pyplot as plt
import matplotlib
from typing import Dict, Any, List, Tuple
from datetime import datetime
from fpdf import FPDF

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("logs/utils.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("utils")

# Make sure matplotlib uses a non-interactive backend
matplotlib.use('Agg')

def create_directories():
    """Create necessary directories if they don't exist."""
    dirs = ["data", "logs", "reports", "models"]
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)
    logger.info("Created necessary directories")

def save_dataframe(df: pd.DataFrame, filename: str, format: str = "csv") -> str:
    """
    Save DataFrame to a file.
    
    Args:
        df: DataFrame to save
        filename: Base filename without extension
        format: Format to save ("csv" or "json")
    
    Returns:
        Path to the saved file
    """
    create_directories()
    
    # Create a valid path
    if not filename.endswith(f".{format}"):
        filename = f"{filename}.{format}"
    
    path = os.path.join("data", filename)
    
    try:
        if format.lower() == "csv":
            df.to_csv(path, index=True)
        elif format.lower() == "json":
            df.to_json(path, orient="records", date_format="iso")
        else:
            logger.error(f"Unsupported format: {format}")
            return ""
        
        logger.info(f"Saved DataFrame to {path}")
        return path
    
    except Exception as e:
        logger.error(f"Error saving DataFrame: {e}")
        return ""

def load_dataframe(path: str) -> pd.DataFrame:
    """
    Load DataFrame from a file.
    
    Args:
        path: Path to the file
    
    Returns:
        Loaded DataFrame
    """
    try:
        if path.endswith(".csv"):
            df = pd.read_csv(path)
        elif path.endswith(".json"):
            df = pd.read_json(path)
        else:
            logger.error(f"Unsupported file format: {path}")
            return pd.DataFrame()
        
        logger.info(f"Loaded DataFrame from {path}")
        return df
    
    except Exception as e:
        logger.error(f"Error loading DataFrame: {e}")
        return pd.DataFrame()

def get_downloadable_link(df: pd.DataFrame, format: str = "csv") -> str:
    """
    Get a base64-encoded link for downloading a DataFrame.
    
    Args:
        df: DataFrame to download
        format: Format to download ("csv" or "json")
    
    Returns:
        Base64-encoded data URI
    """
    try:
        if format.lower() == "csv":
            data = df.to_csv(index=True)
            b64 = base64.b64encode(data.encode()).decode()
            mime = "text/csv"
        elif format.lower() == "json":
            data = df.to_json(orient="records", date_format="iso")
            b64 = base64.b64encode(data.encode()).decode()
            mime = "application/json"
        else:
            logger.error(f"Unsupported format: {format}")
            return ""
        
        href = f"data:{mime};base64,{b64}"
        return href
    
    except Exception as e:
        logger.error(f"Error creating downloadable link: {e}")
        return ""

def plot_time_series(original: pd.Series, cleaned: pd.Series, title: str = "Original vs Cleaned Data") -> plt.Figure:
    """
    Create a plot comparing original and cleaned time series.
    
    Args:
        original: Original time series
        cleaned: Cleaned time series
        title: Plot title
    
    Returns:
        Matplotlib figure
    """
    try:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot original data
        ax.plot(original.index, original.values, 'o-', alpha=0.5, label='Original', color='blue')
        
        # Plot cleaned data
        ax.plot(cleaned.index, cleaned.values, 'o-', alpha=0.8, label='Cleaned', color='green')
        
        # Add labels and legend
        ax.set_title(title)
        ax.set_xlabel('Time')
        ax.set_ylabel('Value')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    except Exception as e:
        logger.error(f"Error creating plot: {e}")
        # Return a blank figure
        return plt.figure()

def plot_anomalies(series: pd.Series, anomaly_mask: pd.Series, title: str = "Detected Anomalies") -> plt.Figure:
    """
    Create a plot highlighting anomalies in a time series.
    
    Args:
        series: Time series data
        anomaly_mask: Boolean mask of anomalies
        title: Plot title
    
    Returns:
        Matplotlib figure
    """
    try:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot all data points
        ax.plot(series.index, series.values, 'o-', alpha=0.5, label='Normal', color='blue')
        
        # Highlight anomalies
        anomalies = series[anomaly_mask]
        if not anomalies.empty:
            ax.scatter(anomalies.index, anomalies.values, color='red', s=80, label='Anomaly', zorder=3)
        
        # Add labels and legend
        ax.set_title(title)
        ax.set_xlabel('Time')
        ax.set_ylabel('Value')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    except Exception as e:
        logger.error(f"Error creating anomaly plot: {e}")
        # Return a blank figure
        return plt.figure()

def generate_report(original_df: pd.DataFrame, cleaned_df: pd.DataFrame, metadata: Dict[str, Any], 
                    job_id: str, output_path: str) -> None:
    """
    Generate a PDF report summarizing the cleaning results.
    
    Args:
        original_df: Original DataFrame before cleaning
        cleaned_df: Cleaned DataFrame after processing
        metadata: Metadata about the cleaning process
        job_id: Unique job identifier
        output_path: Path to save the PDF report
    """
    try:
        logger.info(f"Generating report for job {job_id}")
        
        # Create PDF document
        pdf = FPDF()
        pdf.add_page()
        
        # Set up fonts
        pdf.set_font("Arial", "B", 16)
        
        # Title
        pdf.cell(0, 10, "Time Series Cleaning Report", ln=True, align="C")
        pdf.set_font("Arial", "", 12)
        pdf.cell(0, 10, f"Job ID: {job_id}", ln=True, align="C")
        pdf.cell(0, 10, f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True, align="C")
        pdf.ln(10)
        
        # Cleaning method details
        pdf.set_font("Arial", "B", 14)
        pdf.cell(0, 10, "Cleaning Method", ln=True)
        pdf.set_font("Arial", "", 12)
        
        method = metadata.get("method", "unknown")
        pdf.cell(0, 10, f"Method: {method.capitalize()}", ln=True)
        
        # Parameters section
        if "parameters" in metadata:
            pdf.cell(0, 10, "Parameters:", ln=True)
            params = metadata["parameters"]
            for key, value in params.items():
                pdf.cell(0, 10, f"- {key}: {value}", ln=True)
        pdf.ln(5)
        
        # Results summary
        pdf.set_font("Arial", "B", 14)
        pdf.cell(0, 10, "Cleaning Results", ln=True)
        pdf.set_font("Arial", "", 12)
        
        pdf.cell(0, 10, f"Anomalies Detected: {metadata.get('anomalies_detected', 0)}", ln=True)
        pdf.cell(0, 10, f"Missing Values Filled: {metadata.get('missing_values_filled', 0)}", ln=True)
        
        if "execution_time_ms" in metadata:
            exec_time = metadata["execution_time_ms"]
            pdf.cell(0, 10, f"Execution Time: {exec_time} ms ({exec_time/1000:.2f} seconds)", ln=True)
        pdf.ln(5)
        
        # Create visualization of original vs cleaned data
        try:
            # Find a numeric column to visualize
            numeric_cols = original_df.select_dtypes(include=[np.number]).columns
            # Skip anomaly indicator columns
            numeric_cols = [col for col in numeric_cols if not col.endswith('_anomaly')]
            
            if len(numeric_cols) > 0:
                # Select the first numeric column
                col_to_plot = numeric_cols[0]
                
                # Create figure
                plt.figure(figsize=(10, 6))
                plt.plot(original_df.index, original_df[col_to_plot], 'o-', alpha=0.7, label='Original')
                plt.plot(cleaned_df.index, cleaned_df[col_to_plot], 'o-', alpha=0.7, label='Cleaned')
                plt.title(f'Original vs Cleaned: {col_to_plot}')
                plt.xlabel('Index')
                plt.ylabel('Value')
                plt.legend()
                plt.grid(True, alpha=0.3)
                
                # Save figure to temporary file
                temp_fig_path = f"reports/temp_fig_{job_id}.png"
                plt.savefig(temp_fig_path, dpi=100, bbox_inches='tight')
                plt.close()
                
                # Add figure to PDF
                pdf.ln(5)
                pdf.set_font("Arial", "B", 14)
                pdf.cell(0, 10, "Data Visualization", ln=True)
                pdf.image(temp_fig_path, x=10, y=None, w=180)
                pdf.ln(85)  # Make space for the image
                
                # Clean up temporary file
                try:
                    os.remove(temp_fig_path)
                except:
                    pass
                
        except Exception as e:
            logger.error(f"Error creating visualization: {e}")
            pdf.cell(0, 10, "Error creating visualization", ln=True)
        
        # Statistics comparison
        pdf.set_font("Arial", "B", 14)
        pdf.cell(0, 10, "Statistics Comparison", ln=True)
        pdf.set_font("Arial", "", 12)
        
        # Create a statistics table
        try:
            # Calculate statistics for numeric columns
            numeric_cols = original_df.select_dtypes(include=[np.number]).columns
            # Skip anomaly indicator columns
            numeric_cols = [col for col in numeric_cols if not col.endswith('_anomaly')]
            
            if len(numeric_cols) > 0:
                # Select the first column for detailed stats
                col_to_analyze = numeric_cols[0]
                
                # Calculate statistics
                orig_mean = original_df[col_to_analyze].mean()
                clean_mean = cleaned_df[col_to_analyze].mean()
                orig_std = original_df[col_to_analyze].std()
                clean_std = cleaned_df[col_to_analyze].std()
                orig_min = original_df[col_to_analyze].min()
                clean_min = cleaned_df[col_to_analyze].min()
                orig_max = original_df[col_to_analyze].max()
                clean_max = cleaned_df[col_to_analyze].max()
                
                # Table header
                pdf.cell(60, 10, "Statistic", 1)
                pdf.cell(60, 10, "Original Data", 1)
                pdf.cell(60, 10, "Cleaned Data", 1, ln=True)
                
                # Table rows
                pdf.cell(60, 10, "Mean", 1)
                pdf.cell(60, 10, f"{orig_mean:.4f}", 1)
                pdf.cell(60, 10, f"{clean_mean:.4f}", 1, ln=True)
                
                pdf.cell(60, 10, "Standard Deviation", 1)
                pdf.cell(60, 10, f"{orig_std:.4f}", 1)
                pdf.cell(60, 10, f"{clean_std:.4f}", 1, ln=True)
                
                pdf.cell(60, 10, "Minimum", 1)
                pdf.cell(60, 10, f"{orig_min:.4f}", 1)
                pdf.cell(60, 10, f"{clean_min:.4f}", 1, ln=True)
                
                pdf.cell(60, 10, "Maximum", 1)
                pdf.cell(60, 10, f"{orig_max:.4f}", 1)
                pdf.cell(60, 10, f"{clean_max:.4f}", 1, ln=True)
                
                # Missing values
                orig_missing = original_df[col_to_analyze].isna().sum()
                clean_missing = cleaned_df[col_to_analyze].isna().sum()
                
                pdf.cell(60, 10, "Missing Values", 1)
                pdf.cell(60, 10, f"{orig_missing}", 1)
                pdf.cell(60, 10, f"{clean_missing}", 1, ln=True)
        except Exception as e:
            logger.error(f"Error creating statistics table: {e}")
            pdf.cell(0, 10, "Error creating statistics table", ln=True)
        
        # Method-specific information
        pdf.ln(10)
        pdf.set_font("Arial", "B", 14)
        pdf.cell(0, 10, "Method Details", ln=True)
        pdf.set_font("Arial", "", 12)
        
        if method == "quantum":
            if "quantum_circuit_depth" in metadata:
                pdf.cell(0, 10, f"Quantum Circuit Depth: {metadata.get('quantum_circuit_depth')}", ln=True)
            if "simulator_shots" in metadata:
                pdf.cell(0, 10, f"Simulator Shots: {metadata.get('simulator_shots')}", ln=True)
                
        elif method == "deep_learning":
            if "model_size" in metadata:
                pdf.cell(0, 10, f"Model Size: {metadata.get('model_size')}", ln=True)
            if "training_epochs" in metadata:
                pdf.cell(0, 10, f"Training Epochs: {metadata.get('training_epochs')}", ln=True)
        
        # Save the PDF
        pdf.output(output_path)
        logger.info(f"Report generated successfully: {output_path}")
        
    except Exception as e:
        logger.error(f"Error generating report: {e}")
        # Create a simple error report
        try:
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", "B", 16)
            pdf.cell(0, 10, "Error Generating Report", ln=True, align="C")
            pdf.set_font("Arial", "", 12)
            pdf.cell(0, 10, f"Job ID: {job_id}", ln=True)
            pdf.cell(0, 10, f"Error: {str(e)}", ln=True)
            pdf.output(output_path)
        except:
            logger.error("Failed to create even the error report")
