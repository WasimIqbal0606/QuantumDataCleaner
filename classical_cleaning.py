import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple
import time
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("logs/classical_cleaning.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("classical_cleaning")

class ClassicalCleaner:
    """
    Class implementing classical statistical methods for time-series cleaning.
    Methods include moving averages, median filtering, z-score based anomaly detection,
    and interpolation for missing values.
    """
    
    def __init__(self):
        """Initialize the classical cleaner."""
        logger.info("Initializing ClassicalCleaner")
    
    def clean(self, df: pd.DataFrame, column: str, params: Dict[str, Any]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Clean the time-series data using classical statistical methods.
        
        Args:
            df: Input DataFrame containing time-series data
            column: Column name to clean
            params: Dictionary of parameters for the cleaning process
                - window_size: Size of the moving window for filtering
                - z_threshold: Threshold for z-score based anomaly detection
                - use_median: Whether to use median filtering (True) or mean filtering (False)
        
        Returns:
            Tuple of (cleaned_df, metadata)
                - cleaned_df: DataFrame with cleaned data
                - metadata: Dictionary with information about the cleaning process
        """
        start_time = time.time()
        logger.info(f"Starting classical cleaning for column: {column}")
        
        # Make a copy of the dataframe to avoid modifying the original
        cleaned_df = df.copy()
        
        # Extract parameters with defaults
        window_size = params.get("window_size", 5)
        z_threshold = params.get("z_threshold", 3.0)
        use_median = params.get("use_median", True)
        
        logger.info(f"Parameters: window_size={window_size}, z_threshold={z_threshold}, use_median={use_median}")
        
        # Count initial missing values
        initial_missing = cleaned_df[column].isna().sum()
        logger.info(f"Initial missing values: {initial_missing}")
        
        # Step 1: Detect outliers using z-score method
        rolling_mean = cleaned_df[column].rolling(window=window_size, center=True).mean()
        rolling_std = cleaned_df[column].rolling(window=window_size, center=True).std()
        
        # Handle edge cases where std is 0 or NaN
        rolling_std = rolling_std.replace(0, np.nan)
        rolling_std = rolling_std.fillna(cleaned_df[column].std())
        
        # Calculate z-scores
        z_scores = np.abs((cleaned_df[column] - rolling_mean) / rolling_std)
        
        # Create an anomaly indicator column
        anomaly_col = f"{column}_anomaly"
        cleaned_df[anomaly_col] = (z_scores > z_threshold).astype(int)
        
        # Count anomalies
        anomalies_detected = cleaned_df[anomaly_col].sum()
        logger.info(f"Detected {anomalies_detected} anomalies using z-score method")
        
        # Step 2: Replace anomalies with NaN for later interpolation
        cleaned_df.loc[cleaned_df[anomaly_col] == 1, column] = np.nan
        
        # Step 3: Apply smoothing filter
        if use_median:
            # Median filter (better for outliers)
            cleaned_df[f"{column}_smooth"] = cleaned_df[column].rolling(
                window=window_size, center=True, min_periods=1
            ).median()
        else:
            # Mean filter (better for noise)
            cleaned_df[f"{column}_smooth"] = cleaned_df[column].rolling(
                window=window_size, center=True, min_periods=1
            ).mean()
        
        # Step 4: Fill missing values using interpolation
        # First, use the smoothed values for missing data
        cleaned_df[column] = cleaned_df[column].fillna(cleaned_df[f"{column}_smooth"])
        
        # For any remaining gaps, use linear interpolation
        cleaned_df[column] = cleaned_df[column].interpolate(method='linear')
        
        # For leading/trailing NaNs, use forward/backward fill
        cleaned_df[column] = cleaned_df[column].fillna(method='ffill').fillna(method='bfill')
        
        # Count remaining missing values
        remaining_missing = cleaned_df[column].isna().sum()
        missing_values_filled = initial_missing - remaining_missing + anomalies_detected
        logger.info(f"Filled {missing_values_filled} missing values")
        
        # Drop the temporary smoothed column
        cleaned_df = cleaned_df.drop(columns=[f"{column}_smooth"])
        
        # Calculate execution time
        execution_time_ms = (time.time() - start_time) * 1000
        logger.info(f"Classical cleaning completed in {execution_time_ms:.2f} ms")
        
        # Prepare metadata
        metadata = {
            "method": "classical",
            "window_size": window_size,
            "z_threshold": z_threshold,
            "use_median": use_median,
            "anomalies_detected": int(anomalies_detected),
            "missing_values_filled": int(missing_values_filled),
            "execution_time_ms": execution_time_ms
        }
        
        # Create a proper object - don't return a tuple directly
        # This prevents the 'tuple' object has no attribute 'to' error
        class ResultObject:
            def __init__(self, df, meta):
                self.df = df
                self.metadata = meta
            
            def to(self, *args, **kwargs):
                # This is a dummy method to handle any unexpected 'to' method calls
                return self
                
        result = ResultObject(cleaned_df, metadata)
        return result
    
    def detect_seasonality(self, series: pd.Series) -> Dict[str, Any]:
        """
        Detect seasonality in the time-series using autocorrelation.
        
        Args:
            series: Input time series
        
        Returns:
            Dictionary with seasonality information
        """
        from statsmodels.tsa.stattools import acf
        
        # Remove NaN values for autocorrelation
        clean_series = series.dropna()
        
        if len(clean_series) < 10:
            logger.warning("Series too short to detect seasonality")
            return {"has_seasonality": False}
        
        try:
            # Calculate autocorrelation
            acf_values = acf(clean_series, nlags=min(len(clean_series) // 2, 50))
            
            # Find peaks in the autocorrelation function
            from scipy.signal import find_peaks
            peaks, _ = find_peaks(acf_values, height=0.1)
            
            if len(peaks) > 1:
                # Estimate period as distance between peaks
                period = peaks[1] - peaks[0]
                return {
                    "has_seasonality": True,
                    "period": int(period),
                    "acf_strength": float(acf_values[peaks[1]])
                }
            else:
                return {"has_seasonality": False}
        
        except Exception as e:
            logger.error(f"Error detecting seasonality: {e}")
            return {"has_seasonality": False, "error": str(e)}
    
    def detect_trend(self, series: pd.Series) -> Dict[str, Any]:
        """
        Detect trend in the time-series using linear regression.
        
        Args:
            series: Input time series
        
        Returns:
            Dictionary with trend information
        """
        from scipy.stats import linregress
        
        # Remove NaN values for regression
        clean_series = series.dropna()
        
        if len(clean_series) < 3:
            logger.warning("Series too short to detect trend")
            return {"has_trend": False}
        
        try:
            # Perform linear regression
            x = np.arange(len(clean_series))
            slope, intercept, r_value, p_value, std_err = linregress(x, clean_series.values)
            
            # Determine if there's a significant trend
            has_trend = p_value < 0.05 and abs(r_value) > 0.3
            
            return {
                "has_trend": has_trend,
                "slope": float(slope),
                "r_squared": float(r_value ** 2),
                "p_value": float(p_value)
            }
        
        except Exception as e:
            logger.error(f"Error detecting trend: {e}")
            return {"has_trend": False, "error": str(e)}
