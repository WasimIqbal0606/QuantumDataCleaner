import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, List
import time
import logging
import random

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("logs/bandit_selector.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("bandit_selector")

class BanditSelector:
    """
    Class implementing a multi-armed bandit algorithm to select the best
    cleaning method based on data characteristics.
    """
    
    def __init__(self):
        """Initialize the bandit selector."""
        logger.info("Initializing BanditSelector")
        
        # Available methods
        self.methods = ["classical", "deep", "quantum"]
        
        # Initialize method stats
        self.method_successes = {method: 0 for method in self.methods}
        self.method_trials = {method: 0 for method in self.methods}
        
        # Context history
        self.contexts = []
        self.selected_methods = []
        self.rewards = []
    
    def extract_features(self, df: pd.DataFrame, column: str) -> Dict[str, float]:
        """
        Extract features from the time series to use as context.
        
        Args:
            df: Input DataFrame containing time-series data
            column: Column name to analyze
        
        Returns:
            Dictionary of features
        """
        logger.info(f"Extracting features from column: {column}")
        
        # Get the series
        series = df[column].copy()
        
        # Basic statistics
        features = {}
        
        try:
            # Length and missing values
            features["length"] = len(series)
            features["missing_ratio"] = series.isna().mean()
            
            # Clean series for calculations
            clean_series = series.dropna()
            
            if len(clean_series) > 2:
                # Statistical features
                features["mean"] = clean_series.mean()
                features["std"] = clean_series.std()
                features["skew"] = clean_series.skew()
                features["kurt"] = clean_series.kurt()
                
                # Range features
                features["min"] = clean_series.min()
                features["max"] = clean_series.max()
                features["range"] = features["max"] - features["min"]
                
                # Outlier features using IQR
                q1 = clean_series.quantile(0.25)
                q3 = clean_series.quantile(0.75)
                iqr = q3 - q1
                outlier_thresh = 1.5 * iqr
                outliers = ((clean_series < (q1 - outlier_thresh)) | (clean_series > (q3 + outlier_thresh))).mean()
                features["outlier_ratio"] = outliers
                
                # Trend features
                if len(clean_series) > 5:
                    # Autocorrelation
                    try:
                        # Lag-1 autocorrelation
                        lag1_corr = clean_series.autocorr(lag=1)
                        features["autocorr_lag1"] = lag1_corr if not np.isnan(lag1_corr) else 0
                        
                        # Trend detection
                        x = np.arange(len(clean_series))
                        p = np.polyfit(x, clean_series.values, 1)
                        features["trend_slope"] = p[0]
                    except:
                        features["autocorr_lag1"] = 0
                        features["trend_slope"] = 0
                else:
                    features["autocorr_lag1"] = 0
                    features["trend_slope"] = 0
                
                # Volatility features
                if len(clean_series) > 3:
                    # Ratio of differences to mean
                    diffs = clean_series.diff().dropna()
                    features["volatility"] = diffs.abs().mean() / (features["mean"] if features["mean"] != 0 else 1)
                else:
                    features["volatility"] = 0
            else:
                # Default values for short series
                features["mean"] = series.mean() if not series.isna().all() else 0
                features["std"] = 0
                features["skew"] = 0
                features["kurt"] = 0
                features["min"] = features["mean"]
                features["max"] = features["mean"]
                features["range"] = 0
                features["outlier_ratio"] = 0
                features["autocorr_lag1"] = 0
                features["trend_slope"] = 0
                features["volatility"] = 0
        
        except Exception as e:
            logger.error(f"Error extracting features: {e}")
            # Set default features
            features = {
                "length": len(series),
                "missing_ratio": series.isna().mean(),
                "mean": 0,
                "std": 0,
                "skew": 0,
                "kurt": 0,
                "min": 0,
                "max": 0,
                "range": 0,
                "outlier_ratio": 0,
                "autocorr_lag1": 0,
                "trend_slope": 0,
                "volatility": 0
            }
        
        logger.info(f"Extracted features: {features}")
        return features
    
    def select_method(self, context: Dict[str, float]) -> str:
        """
        Select a cleaning method using contextual bandit strategy.
        
        Args:
            context: Dictionary of features describing the time series
        
        Returns:
            Selected method name
        """
        logger.info("Selecting method using contextual bandit")
        
        # Calculate similarity with previous contexts if available
        if self.contexts and random.random() < 0.8:  # 80% chance to use history
            # Calculate similarities with previous contexts
            similarities = []
            for prev_context in self.contexts:
                sim = self.calculate_similarity(context, prev_context)
                similarities.append(sim)
            
            # Find most similar contexts
            top_k = 3
            if len(similarities) > top_k:
                top_indices = np.argsort(similarities)[-top_k:]
                
                # Get average rewards for each method from top similar contexts
                method_weights = {method: 0 for method in self.methods}
                
                for idx in top_indices:
                    method = self.selected_methods[idx]
                    reward = self.rewards[idx]
                    method_weights[method] += reward * similarities[idx]
                
                # Choose the method with highest weighted reward (exploitation)
                if random.random() < 0.8:  # 80% exploitation, 20% exploration
                    best_method = max(method_weights.items(), key=lambda x: x[1])[0]
                    logger.info(f"Selected method based on similar context: {best_method}")
                    return best_method
        
        # Epsilon-greedy strategy for exploration
        if random.random() < 0.2:  # 20% exploration
            selected_method = random.choice(self.methods)
            logger.info(f"Exploration: randomly selected {selected_method}")
            return selected_method
        
        # Use UCB1 formula for exploitation
        ucb_values = {}
        total_trials = sum(self.method_trials.values()) or 1
        
        for method in self.methods:
            trials = self.method_trials[method] or 1
            successes = self.method_successes[method]
            
            # UCB1 formula: average reward + exploration bonus
            success_rate = successes / trials
            exploration_term = np.sqrt(2 * np.log(total_trials) / trials)
            ucb = success_rate + exploration_term
            
            # For contextual bandits, adjust UCB based on context features
            # For complex time series with high volatility, favor deep learning
            if context["volatility"] > 0.3 and context["outlier_ratio"] > 0.1:
                if method == "deep":
                    ucb *= 1.2
            
            # For series with many missing values, favor classical methods
            if context["missing_ratio"] > 0.3:
                if method == "classical":
                    ucb *= 1.1
            
            # For series with strong autocorrelation, favor quantum methods
            if abs(context["autocorr_lag1"]) > 0.7:
                if method == "quantum":
                    ucb *= 1.1
            
            ucb_values[method] = ucb
        
        # Select the method with the highest UCB value
        selected_method = max(ucb_values.items(), key=lambda x: x[1])[0]
        logger.info(f"UCB selection: {selected_method} (UCB values: {ucb_values})")
        
        return selected_method
    
    def update_reward(self, method: str, context: Dict[str, float], reward: float) -> None:
        """
        Update the bandit's knowledge with the reward received.
        
        Args:
            method: The method that was used
            context: The context in which the method was used
            reward: The reward received (higher is better)
        """
        logger.info(f"Updating bandit with reward {reward} for method {method}")
        
        # Update method statistics
        self.method_trials[method] += 1
        self.method_successes[method] += reward
        
        # Store context, method, and reward for future similarity calculations
        self.contexts.append(context)
        self.selected_methods.append(method)
        self.rewards.append(reward)
    
    def calculate_similarity(self, context1: Dict[str, float], context2: Dict[str, float]) -> float:
        """
        Calculate similarity between two contexts.
        
        Args:
            context1: First context
            context2: Second context
        
        Returns:
            Similarity score (0 to 1)
        """
        # Get common keys
        common_keys = set(context1.keys()) & set(context2.keys())
        
        if not common_keys:
            return 0
        
        # Calculate Euclidean distance on normalized features
        squared_diff_sum = 0
        for key in common_keys:
            # Skip non-numeric or zero-range features
            if not isinstance(context1[key], (int, float)) or not isinstance(context2[key], (int, float)):
                continue
            
            # Normalize values to [0, 1] range based on typical ranges
            if key == "length":
                # Normalize log length (1 to 1M points)
                val1 = np.log1p(context1[key]) / np.log1p(1e6)
                val2 = np.log1p(context2[key]) / np.log1p(1e6)
            elif key in ["missing_ratio", "outlier_ratio"]:
                # Already in [0, 1] range
                val1 = context1[key]
                val2 = context2[key]
            elif key in ["autocorr_lag1"]:
                # In [-1, 1] range
                val1 = (context1[key] + 1) / 2
                val2 = (context2[key] + 1) / 2
            elif key in ["skew", "kurt"]:
                # Normalize with sigmoid-like function
                val1 = 1 / (1 + np.exp(-context1[key] / 2))
                val2 = 1 / (1 + np.exp(-context2[key] / 2))
            elif key == "volatility":
                # Normalize with cap at 2
                val1 = min(context1[key], 2) / 2
                val2 = min(context2[key], 2) / 2
            else:
                # General case - use relative difference
                # This handles mean, std, min, max, range, trend_slope
                combined_abs = abs(context1[key]) + abs(context2[key])
                if combined_abs > 0:
                    diff = abs(context1[key] - context2[key]) / combined_abs
                    val1, val2 = 0, diff  # Just need the difference
                else:
                    val1, val2 = 0, 0  # Both are zero, so no difference
            
            squared_diff_sum += (val1 - val2) ** 2
        
        # Convert distance to similarity score
        distance = np.sqrt(squared_diff_sum / len(common_keys))
        similarity = 1 / (1 + distance)  # Convert to similarity (1 is most similar)
        
        return similarity
    
    def evaluate_method_fit(self, method: str, context: Dict[str, float]) -> float:
        """
        Evaluate how well a method fits the given context.
        
        Args:
            method: The method to evaluate
            context: The context to evaluate the method for
        
        Returns:
            Fitness score (0 to 1)
        """
        # Default score
        score = 0.5
        
        # Method-specific scoring based on context
        if method == "classical":
            # Classical methods are good for:
            # - Small to medium datasets
            # - Low to medium volatility
            # - Few outliers
            # - Simple patterns
            
            size_score = 1.0 - min(context["length"], 10000) / 10000
            volatility_score = 1.0 - min(context["volatility"], 1.0)
            outlier_score = 1.0 - min(context["outlier_ratio"], 0.5) * 2
            complexity_score = 1.0 - min(abs(context["autocorr_lag1"]), 0.8) * 1.25
            
            score = 0.25 * size_score + 0.3 * volatility_score + 0.25 * outlier_score + 0.2 * complexity_score
        
        elif method == "deep":
            # Deep learning methods are good for:
            # - Medium to large datasets
            # - Complex patterns
            # - High autocorrelation
            # - Moderate to high volatility
            
            size_score = min(context["length"], 5000) / 5000
            pattern_score = min(abs(context["autocorr_lag1"]), 1.0)
            complexity_score = min(context["kurt"] + 3, 6) / 6  # Kurtosis as a proxy for complexity
            volatility_score = min(context["volatility"], 1.0)
            
            score = 0.3 * size_score + 0.3 * pattern_score + 0.2 * complexity_score + 0.2 * volatility_score
        
        elif method == "quantum":
            # Quantum methods are good for:
            # - Detecting subtle patterns
            # - Handling high kurtosis data
            # - Lower predictability (lower autocorrelation)
            # - Complex distributions (high skew/kurtosis)
            
            unpredictability = 1.0 - min(abs(context["autocorr_lag1"]), 0.9)
            complexity_score = min(abs(context["skew"]), 2) / 2
            kurtosis_score = min(abs(context["kurt"]), 5) / 5
            outlier_score = min(context["outlier_ratio"], 0.5) * 2
            
            score = 0.3 * unpredictability + 0.2 * complexity_score + 0.2 * kurtosis_score + 0.3 * outlier_score
        
        # Ensure score is between 0 and 1
        return max(0, min(1, score))
    
    def select_best_method(self, df: pd.DataFrame, column: str) -> Tuple[str, Dict[str, Any]]:
        """
        Select the best cleaning method for the given data.
        
        Args:
            df: Input DataFrame containing time-series data
            column: Column name to analyze
        
        Returns:
            Tuple of (selected_method, method_params)
        """
        logger.info(f"Selecting best method for column: {column}")
        
        # Extract context features
        context = self.extract_features(df, column)
        
        # Select method using bandit strategy
        selected_method = self.select_method(context)
        
        # Set default parameters for the selected method
        method_params = {}
        
        if selected_method == "classical":
            method_params = {
                "window_size": 5,
                "z_threshold": 3.0,
                "use_median": True
            }
        elif selected_method == "deep":
            # Adjust sequence length based on dataset size
            seq_length = min(30, max(10, len(df) // 10))
            
            method_params = {
                "sequence_length": seq_length,
                "epochs": 50,
                "learning_rate": 0.001
            }
        elif selected_method == "quantum":
            method_params = {
                "shots": 500,
                "layers": 2,
                "simulator": True
            }
        
        # Adjust parameters based on context
        self.adjust_params(selected_method, method_params, context)
        
        logger.info(f"Selected method: {selected_method} with params: {method_params}")
        
        # Simulate getting a reward (in a real scenario, this would come after cleaning)
        method_fit = self.evaluate_method_fit(selected_method, context)
        logger.info(f"Method fitness score: {method_fit}")
        
        # Update bandit knowledge
        self.update_reward(selected_method, context, method_fit)
        
        return selected_method, method_params
    
    def adjust_params(self, method: str, params: Dict[str, Any], context: Dict[str, float]) -> None:
        """
        Adjust method parameters based on context.
        
        Args:
            method: The selected method
            params: Parameters to adjust
            context: Context features
        """
        if method == "classical":
            # Adjust window size based on series length and volatility
            if context["length"] > 1000:
                params["window_size"] = max(5, min(21, int(np.sqrt(context["length"]) / 10)))
            
            # Adjust z-threshold based on outlier ratio
            if context["outlier_ratio"] > 0.1:
                params["z_threshold"] = max(2.0, 3.0 - context["outlier_ratio"] * 5)
            
            # Use median for high-outlier data
            params["use_median"] = context["outlier_ratio"] > 0.05
        
        elif method == "deep":
            # Adjust sequence length based on autocorrelation
            if abs(context["autocorr_lag1"]) > 0.7:
                # For highly correlated data, use longer sequences
                params["sequence_length"] = min(50, max(10, int(context["length"] / 8)))
            
            # Adjust epochs based on data size
            if context["length"] > 1000:
                params["epochs"] = min(100, max(20, int(50 * np.sqrt(context["length"] / 1000))))
            
            # Adjust learning rate based on volatility
            if context["volatility"] > 0.5:
                params["learning_rate"] = 0.0005  # Lower learning rate for volatile data
        
        elif method == "quantum":
            # Adjust layers based on complexity
            complexity = (abs(context["skew"]) + abs(context["kurt"])) / 4
            params["layers"] = max(1, min(5, int(1 + complexity * 2)))
            
            # Adjust shots based on volatility
            if context["volatility"] < 0.1:
                params["shots"] = 200  # Fewer shots needed for stable data
            elif context["volatility"] > 0.5:
                params["shots"] = 1000  # More shots for volatile data
