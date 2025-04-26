import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple
import time
import logging
from quantum_cleaning import ResultObject  # Import the common ResultObject

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
    interpolation for missing values, and advanced techniques like wavelet denoising,
    Fourier filtering, Savitzky-Golay filtering, Hampel filter, etc.
    Also provides optimized handling for large datasets through chunking.
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
            
            def __iter__(self):
                # This makes the object iterable and unpacks like a tuple (df, metadata)
                yield self.df
                yield self.metadata
                
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
            
    def _process_large_dataset(self, df: pd.DataFrame, column: str, params: Dict[str, Any], chunk_size: int) -> pd.DataFrame:
        """
        Process a large dataset in chunks to improve performance.
        
        Args:
            df: Input DataFrame containing time-series data
            column: Column name to clean
            params: Dictionary of parameters for the cleaning process
            chunk_size: Size of each chunk for processing
            
        Returns:
            DataFrame with cleaned data
        """
        logger.info(f"Processing large dataset in chunks of size {chunk_size}")
        
        # Create a result dataframe
        result_df = df.copy()
        
        # Create an empty outlier column if we'll be detecting outliers
        result_df[f"{column}_outlier"] = 0
        
        # Calculate total chunks
        total_rows = len(df)
        total_chunks = (total_rows + chunk_size - 1) // chunk_size
        
        logger.info(f"Processing {total_rows} rows in {total_chunks} chunks")
        
        # Process each chunk with overlap to avoid edge effects
        overlap = params.get("window_size", 5) * 2
        
        for i in range(total_chunks):
            # Calculate chunk boundaries with overlap
            start_idx = max(0, i * chunk_size - overlap if i > 0 else 0)
            end_idx = min(total_rows, (i + 1) * chunk_size + overlap if i < total_chunks - 1 else total_rows)
            
            logger.info(f"Processing chunk {i+1}/{total_chunks}: rows {start_idx}-{end_idx}")
            
            # Extract chunk
            chunk_df = df.iloc[start_idx:end_idx].copy()
            
            # Process the chunk - use simplified method for better performance
            chunk_df = self._process_chunk(chunk_df, column, params)
            
            # Remove the overlap when writing back
            chunk_start = 0 if i == 0 else overlap
            chunk_end = len(chunk_df) if i == total_chunks - 1 else len(chunk_df) - overlap
            
            # Write chunk results back to master dataframe
            actual_start = start_idx + chunk_start
            actual_end = start_idx + chunk_end
            
            # Copy the cleaned data back
            result_df.iloc[actual_start:actual_end, result_df.columns.get_loc(column)] = \
                chunk_df.iloc[chunk_start:chunk_end, chunk_df.columns.get_loc(column)].values
            
            # Copy any outlier markings
            result_df.iloc[actual_start:actual_end, result_df.columns.get_loc(f"{column}_outlier")] = \
                chunk_df.iloc[chunk_start:chunk_end, chunk_df.columns.get_loc(f"{column}_outlier")].values
        
        logger.info("Finished processing all chunks")
        return result_df
    
    def _process_chunk(self, df: pd.DataFrame, column: str, params: Dict[str, Any]) -> pd.DataFrame:
        """
        Process a single chunk of data with simplified method for better performance.
        
        Args:
            df: Input DataFrame containing time-series data (chunk)
            column: Column name to clean
            params: Dictionary of parameters for the cleaning process
            
        Returns:
            DataFrame with cleaned chunk
        """
        # Extract parameters with defaults
        window_size = params.get("window_size", 5)
        z_threshold = params.get("z_threshold", 3.0)
        use_median = params.get("use_median", True)
        
        # Make sure the outlier column exists
        if f"{column}_outlier" not in df.columns:
            df[f"{column}_outlier"] = 0
        
        # Detect outliers using z-score method (simplified for performance)
        if use_median:
            rolling_center = df[column].rolling(window=window_size, center=True).median()
        else:
            rolling_center = df[column].rolling(window=window_size, center=True).mean()
            
        rolling_std = df[column].rolling(window=window_size, center=True).std()
        
        # Handle edge cases where std is 0 or NaN
        rolling_std = rolling_std.replace(0, np.nan)
        rolling_std = rolling_std.fillna(df[column].std())
        
        # Calculate z-scores
        z_scores = np.abs((df[column] - rolling_center) / rolling_std)
        
        # Mark outliers
        df.loc[z_scores > z_threshold, f"{column}_outlier"] = 1
        
        # Replace outliers with NaN for interpolation
        df.loc[z_scores > z_threshold, column] = np.nan
        
        # Interpolate missing values
        df[column] = df[column].interpolate(method='linear').fillna(method='bfill').fillna(method='ffill')
        
        # Apply minimal smoothing
        if use_median:
            smoothed = df[column].rolling(window=3, center=True, min_periods=1).median()
        else:
            smoothed = df[column].rolling(window=3, center=True, min_periods=1).mean()
            
        # Fill any NaNs from the smoothing operation
        smoothed = smoothed.fillna(df[column])
        df[column] = smoothed
        
        return df
    
    def _z_score_outlier_detection(self, df: pd.DataFrame, column: str, window_size: int, 
                                 z_threshold: float, use_median: bool, spikes_only: bool) -> pd.DataFrame:
        """
        Detect outliers using the z-score method.
        
        Args:
            df: Input DataFrame
            column: Column to clean
            window_size: Window size for rolling statistics
            z_threshold: Z-score threshold
            use_median: Whether to use median (True) or mean (False)
            spikes_only: Whether to only detect positive outliers
            
        Returns:
            DataFrame with outliers detected
        """
        # Calculate rolling statistics
        if use_median:
            rolling_center = df[column].rolling(window=window_size, center=True).median()
        else:
            rolling_center = df[column].rolling(window=window_size, center=True).mean()
            
        rolling_std = df[column].rolling(window=window_size, center=True).std()
        
        # Handle edge cases where std is 0 or NaN
        rolling_std = rolling_std.replace(0, np.nan)
        rolling_std = rolling_std.fillna(df[column].std())
        
        # Calculate z-scores
        z_scores = (df[column] - rolling_center) / rolling_std
        
        # Create outlier mask
        if spikes_only:
            # Only detect positive spikes
            outlier_mask = z_scores > z_threshold
        else:
            # Detect both positive and negative outliers
            outlier_mask = np.abs(z_scores) > z_threshold
        
        # Create an outlier indicator column
        outlier_col = f"{column}_outlier"
        df[outlier_col] = outlier_mask.astype(int)
        
        # Replace outliers with NaN
        df.loc[outlier_mask, column] = np.nan
        
        return df
    
    def _iqr_outlier_detection(self, df: pd.DataFrame, column: str, params: Dict[str, Any]) -> pd.DataFrame:
        """
        Detect outliers using the Interquartile Range (IQR) method.
        
        Args:
            df: Input DataFrame
            column: Column to clean
            params: Parameters with IQR multiplier
            
        Returns:
            DataFrame with outliers detected
        """
        # Get parameters
        iqr_multiplier = params.get("iqr_multiplier", 1.5)
        
        # Calculate quartiles
        q1 = df[column].quantile(0.25)
        q3 = df[column].quantile(0.75)
        iqr = q3 - q1
        
        # Define bounds
        lower_bound = q1 - iqr_multiplier * iqr
        upper_bound = q3 + iqr_multiplier * iqr
        
        # Detect outliers
        outlier_mask = (df[column] < lower_bound) | (df[column] > upper_bound)
        
        # Create an outlier indicator column
        outlier_col = f"{column}_outlier"
        df[outlier_col] = outlier_mask.astype(int)
        
        # Replace outliers with NaN
        df.loc[outlier_mask, column] = np.nan
        
        return df
    
    def _hampel_filter(self, df: pd.DataFrame, column: str, window_size: int, threshold: float) -> pd.DataFrame:
        """
        Apply Hampel filter for outlier detection.
        
        Args:
            df: Input DataFrame
            column: Column to clean
            window_size: Window size
            threshold: Threshold multiplier
            
        Returns:
            DataFrame with outliers detected
        """
        # Create a copy of the data
        series = df[column].copy()
        rolling_median = series.rolling(window=window_size, center=True).median()
        
        # Calculate median absolute deviation (MAD)
        rolling_mad = (series - rolling_median).abs().rolling(window=window_size, center=True).median()
        
        # Scale MAD (approximation to standard deviation)
        rolling_mad = rolling_mad * 1.4826
        
        # Avoid division by zero
        rolling_mad = rolling_mad.replace(0, np.nan)
        rolling_mad = rolling_mad.fillna(series.std() * 0.6745)  # Convert std to MAD scale
        
        # Calculate deviation scores
        deviation = np.abs(series - rolling_median) / rolling_mad
        
        # Detect outliers
        outlier_mask = deviation > threshold
        
        # Create an outlier indicator column
        outlier_col = f"{column}_outlier"
        df[outlier_col] = outlier_mask.astype(int)
        
        # Replace outliers with NaN
        df.loc[outlier_mask, column] = np.nan
        
        return df
    
    def _interpolate_missing_values(self, df: pd.DataFrame, column: str, params: Dict[str, Any]) -> pd.DataFrame:
        """
        Interpolate missing values in the time series.
        
        Args:
            df: Input DataFrame
            column: Column to clean
            params: Parameters for interpolation
            
        Returns:
            DataFrame with interpolated values
        """
        # Get interpolation method
        interp_method = params.get("interpolation_method", "linear")
        
        # Different interpolation methods
        if interp_method == "linear":
            df[column] = df[column].interpolate(method='linear')
        elif interp_method == "spline":
            df[column] = df[column].interpolate(method='spline', order=3)
        elif interp_method == "polynomial":
            df[column] = df[column].interpolate(method='polynomial', order=2)
        elif interp_method == "nearest":
            df[column] = df[column].interpolate(method='nearest')
        else:
            # Default to linear
            df[column] = df[column].interpolate(method='linear')
        
        # Fill any remaining NaNs at the edges
        df[column] = df[column].fillna(method='ffill').fillna(method='bfill')
        
        return df
    
    def _apply_savgol_filter(self, df: pd.DataFrame, column: str, window_size: int, order: int) -> pd.DataFrame:
        """
        Apply Savitzky-Golay filter for smoothing.
        
        Args:
            df: Input DataFrame
            column: Column to clean
            window_size: Window size (must be odd)
            order: Polynomial order
            
        Returns:
            DataFrame with smoothed values
        """
        try:
            from scipy.signal import savgol_filter
            
            # Make sure window size is odd
            if window_size % 2 == 0:
                window_size += 1
            
            # Apply filter
            series = df[column].copy()
            
            # Temporarily fill NaNs for filtering
            temp_series = series.fillna(method='ffill').fillna(method='bfill')
            
            # Apply filter
            filtered = savgol_filter(temp_series, window_size, order)
            
            # Create a smoothed version of the column
            df[f"{column}_savgol"] = filtered
            
            # Use filtered values to replace NaNs in the original
            mask = df[column].isna()
            df.loc[mask, column] = df.loc[mask, f"{column}_savgol"]
            
            # Drop temporary column
            df = df.drop(columns=[f"{column}_savgol"])
            
        except Exception as e:
            logger.error(f"Error applying Savitzky-Golay filter: {e}")
        
        return df
    
    def _wavelet_denoising(self, df: pd.DataFrame, column: str) -> pd.DataFrame:
        """
        Apply wavelet transform denoising.
        
        Args:
            df: Input DataFrame
            column: Column to clean
            
        Returns:
            DataFrame with denoised values
        """
        try:
            import pywt
            
            # Get the data
            series = df[column].copy().fillna(method='ffill').fillna(method='bfill')
            
            # Apply wavelet transform
            coeffs = pywt.wavedec(series, 'db4', level=3)
            
            # Threshold detail coefficients
            for i in range(1, len(coeffs)):
                coeffs[i] = pywt.threshold(coeffs[i], np.std(coeffs[i])/2, 'soft')
            
            # Reconstruct signal
            reconstructed = pywt.waverec(coeffs, 'db4')
            
            # Adjust length if needed
            if len(reconstructed) > len(series):
                reconstructed = reconstructed[:len(series)]
            elif len(reconstructed) < len(series):
                padding = np.array([reconstructed[-1]] * (len(series) - len(reconstructed)))
                reconstructed = np.concatenate([reconstructed, padding])
            
            # Create a denoised version of the column
            df[f"{column}_denoised"] = reconstructed
            
            # Replace NaNs with denoised values
            mask = df[column].isna()
            df.loc[mask, column] = df.loc[mask, f"{column}_denoised"]
            
            # Drop temporary column
            df = df.drop(columns=[f"{column}_denoised"])
            
        except Exception as e:
            logger.error(f"Error applying wavelet denoising: {e}, using fallback method")
            # Fallback to simple smoothing
            if len(df) > 0:
                window_size = min(5, len(df) // 5)
                if window_size > 0:
                    df[column] = df[column].rolling(window=window_size, center=True, min_periods=1).mean()
        
        return df
    
    def _fourier_filtering(self, df: pd.DataFrame, column: str) -> pd.DataFrame:
        """
        Apply Fourier transform filtering to remove high-frequency noise.
        
        Args:
            df: Input DataFrame
            column: Column to clean
            
        Returns:
            DataFrame with filtered values
        """
        try:
            from scipy.fft import fft, ifft
            
            # Get the data
            series = df[column].copy().fillna(method='ffill').fillna(method='bfill')
            values = series.values
            
            # Compute FFT
            fft_values = fft(values)
            
            # Keep only the first 10% of frequencies
            n = len(fft_values)
            cut_off = int(n * 0.1)
            fft_values[cut_off:-cut_off] = 0
            
            # Compute inverse FFT
            filtered = ifft(fft_values)
            
            # Convert back to real values
            filtered = np.real(filtered)
            
            # Create a filtered version of the column
            df[f"{column}_filtered"] = filtered
            
            # Replace NaNs with filtered values
            mask = df[column].isna()
            df.loc[mask, column] = df.loc[mask, f"{column}_filtered"]
            
            # Drop temporary column
            df = df.drop(columns=[f"{column}_filtered"])
            
        except Exception as e:
            logger.error(f"Error applying Fourier filtering: {e}, using fallback method")
            # Fallback to simple smoothing
            if len(df) > 0:
                window_size = min(5, len(df) // 5)
                if window_size > 0:
                    df[column] = df[column].rolling(window=window_size, center=True, min_periods=1).mean()
        
        return df
    
    def _preprocess_time_series(self, df: pd.DataFrame, column: str, trend_removal: bool, seasonal_adjust: bool) -> pd.DataFrame:
        """
        Preprocess time series by removing trend and/or seasonality.
        
        Args:
            df: Input DataFrame
            column: Column to preprocess
            trend_removal: Whether to remove trend
            seasonal_adjust: Whether to adjust for seasonality
            
        Returns:
            DataFrame with preprocessed values
        """
        try:
            # Make a copy
            result_df = df.copy()
            
            # Store original values
            result_df[f"{column}_original"] = result_df[column].copy()
            
            if trend_removal:
                # Detect trend
                trend_info = self.detect_trend(result_df[column])
                
                if trend_info["has_trend"]:
                    # Remove trend
                    x = np.arange(len(result_df))
                    slope = trend_info["slope"]
                    intercept = result_df[column].iloc[0] - slope * x[0]
                    trend = intercept + slope * x
                    
                    # Store trend component
                    result_df[f"{column}_trend"] = trend
                    
                    # Remove trend
                    result_df[column] = result_df[column] - trend
            
            if seasonal_adjust:
                # Detect seasonality
                season_info = self.detect_seasonality(result_df[column])
                
                if season_info["has_seasonality"]:
                    period = season_info["period"]
                    
                    # Calculate seasonal component
                    grouped = result_df[column].groupby(result_df.index % period)
                    seasonal_means = grouped.transform('mean')
                    
                    # Store seasonal component
                    result_df[f"{column}_seasonal"] = seasonal_means
                    
                    # Remove seasonality
                    result_df[column] = result_df[column] - seasonal_means
            
            return result_df
            
        except Exception as e:
            logger.error(f"Error preprocessing time series: {e}")
            return df
