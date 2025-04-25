import os
import uuid
import json
import logging
import chromadb
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional

# Setup logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TimeSeriesDatabase:
    """
    A class for storing, retrieving, and searching time-series data using ChromaDB.
    This enables semantic search and similarity-based retrieval of time-series data.
    """
    
    def __init__(self, persist_directory: str = "db"):
        """
        Initialize the time-series database.
        
        Args:
            persist_directory: Directory where ChromaDB will persist the data
        """
        # Create directory if it doesn't exist
        os.makedirs(persist_directory, exist_ok=True)
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(path=persist_directory)
        logger.info(f"Initialized ChromaDB with persistence directory: {persist_directory}")
        
        # Create collection for time-series data if it doesn't exist
        try:
            self.collection = self.client.get_collection("time_series_data")
            logger.info("Using existing time-series collection")
        except Exception as e:
            logger.info(f"Collection not found, creating new one: {str(e)}")
            self.collection = self.client.create_collection(
                name="time_series_data",
                metadata={"description": "Collection of time-series data with metadata"}
            )
            logger.info("Created new time-series collection")
    
    def time_series_to_embedding(self, series: pd.Series) -> List[float]:
        """
        Convert a time series to a vector embedding.
        This uses statistical features of the time series as the embedding.
        
        Args:
            series: Time series data
            
        Returns:
            Vector embedding representing the time series
        """
        # Calculate statistical features
        features = []
        
        # Basic statistics
        features.append(float(series.mean()))
        features.append(float(series.std()) if len(series) > 1 else 0.0)
        features.append(float(series.min()))
        features.append(float(series.max()))
        
        # Range and volatility
        features.append(float(series.max() - series.min()))
        
        # Autocorrelation (lag 1)
        if len(series) > 1:
            autocorr = series.autocorr(lag=1)
            features.append(float(autocorr) if not pd.isna(autocorr) else 0.0)
        else:
            features.append(0.0)
        
        # Trend estimation (simple linear regression slope)
        if len(series) > 1:
            x = np.arange(len(series))
            y = series.values
            valid_indices = ~np.isnan(y)
            if np.sum(valid_indices) > 1:
                x_valid = x[valid_indices]
                y_valid = y[valid_indices]
                if len(x_valid) > 1:
                    slope = np.polyfit(x_valid, y_valid, 1)[0]
                    features.append(float(slope))
                else:
                    features.append(0.0)
            else:
                features.append(0.0)
        else:
            features.append(0.0)
        
        # Missing data ratio
        features.append(float(series.isna().mean()))
        
        # Skewness and kurtosis (using simple approximations)
        if len(series) > 2:
            try:
                features.append(float(series.skew()))
            except:
                features.append(0.0)
            
            try:
                features.append(float(series.kurtosis()))
            except:
                features.append(0.0)
        else:
            features.extend([0.0, 0.0])
            
        # Ensure all values are float and not numpy types
        features = [float(f) if not pd.isna(f) else 0.0 for f in features]
        
        # Normalize the feature vector
        norm = np.linalg.norm(features)
        if norm > 0:
            features = [f / norm for f in features]
            
        return features
    
    def store_time_series(self, 
                        df: pd.DataFrame, 
                        column: str,
                        metadata: Dict[str, Any],
                        job_id: Optional[str] = None) -> str:
        """
        Store a time series and its metadata in the database.
        
        Args:
            df: DataFrame containing the time series
            column: Column name of the time series
            metadata: Dictionary of metadata about the time series
            job_id: Optional job ID to use, generated if not provided
            
        Returns:
            ID of the stored time series
        """
        # Generate ID if not provided
        if job_id is None:
            job_id = str(uuid.uuid4())
            
        # Extract the time series
        series = df[column].copy()
        
        # Convert the time series to an embedding
        embedding = self.time_series_to_embedding(series)
        
        # Extract key statistics for the document
        stats = {
            "mean": float(series.mean()) if not pd.isna(series.mean()) else 0.0,
            "std": float(series.std()) if not pd.isna(series.std()) else 0.0,
            "min": float(series.min()) if not pd.isna(series.min()) else 0.0,
            "max": float(series.max()) if not pd.isna(series.max()) else 0.0,
            "length": len(series),
            "missing_ratio": float(series.isna().mean()) if not pd.isna(series.isna().mean()) else 0.0
        }
        
        # Create document text (description of the time series)
        document = (
            f"Time series data with {stats['length']} points. "
            f"Mean: {stats['mean']:.2f}, Std: {stats['std']:.2f}, "
            f"Range: {stats['min']:.2f} to {stats['max']:.2f}. "
            f"Missing data: {stats['missing_ratio']*100:.1f}%. "
            f"Description: {metadata.get('description', 'No description')}"
        )
        
        # Prepare metadata for storage
        # Combine provided metadata with statistics and ensure all values are JSON serializable
        combined_metadata = {**metadata, **stats}
        json_metadata = {k: v if isinstance(v, (str, int, float, bool)) else str(v) for k, v in combined_metadata.items()}
        
        # Store in ChromaDB
        try:
            self.collection.add(
                ids=[job_id],
                embeddings=[embedding],
                documents=[document],
                metadatas=[json_metadata]
            )
            logger.info(f"Stored time series with ID: {job_id}")
            return job_id
        except Exception as e:
            logger.error(f"Error storing time series: {str(e)}")
            raise
    
    def get_time_series_metadata(self, job_id: str) -> Dict[str, Any]:
        """
        Retrieve metadata for a stored time series.
        
        Args:
            job_id: ID of the time series
            
        Returns:
            Dictionary of metadata
        """
        try:
            result = self.collection.get(ids=[job_id], include=["metadatas"])
            if result and result["metadatas"]:
                return result["metadatas"][0]
            else:
                logger.warning(f"No metadata found for job ID: {job_id}")
                return {}
        except Exception as e:
            logger.error(f"Error retrieving metadata: {str(e)}")
            return {}
    
    def find_similar_time_series(self, 
                               df: pd.DataFrame, 
                               column: str,
                               n_results: int = 5) -> List[Dict[str, Any]]:
        """
        Find similar time series to the given one.
        
        Args:
            df: DataFrame containing the time series
            column: Column name of the time series
            n_results: Number of similar time series to return
            
        Returns:
            List of dictionaries containing similar time series info
        """
        # Extract the time series
        series = df[column].copy()
        
        # Convert to embedding
        embedding = self.time_series_to_embedding(series)
        
        # Search for similar time series
        try:
            results = self.collection.query(
                query_embeddings=[embedding],
                n_results=n_results,
                include=["metadatas", "documents", "distances"]
            )
            
            # Format the results
            formatted_results = []
            if results and results["ids"]:
                for i, job_id in enumerate(results["ids"][0]):
                    formatted_results.append({
                        "job_id": job_id,
                        "metadata": results["metadatas"][0][i],
                        "description": results["documents"][0][i],
                        "similarity": 1.0 - min(results["distances"][0][i], 1.0)  # Convert distance to similarity score
                    })
            
            return formatted_results
        except Exception as e:
            logger.error(f"Error searching for similar time series: {str(e)}")
            return []
    
    def list_all_time_series(self, limit: int = 100) -> List[Dict[str, Any]]:
        """
        List all stored time series.
        
        Args:
            limit: Maximum number of time series to return
            
        Returns:
            List of dictionaries containing time series info
        """
        try:
            # Get all items from the collection
            results = self.collection.get(limit=limit, include=["metadatas", "documents"])
            
            # Format the results
            formatted_results = []
            if results and results["ids"]:
                for i, job_id in enumerate(results["ids"]):
                    formatted_results.append({
                        "job_id": job_id,
                        "metadata": results["metadatas"][i],
                        "description": results["documents"][i]
                    })
            
            return formatted_results
        except Exception as e:
            logger.error(f"Error listing time series: {str(e)}")
            return []
    
    def delete_time_series(self, job_id: str) -> bool:
        """
        Delete a stored time series.
        
        Args:
            job_id: ID of the time series to delete
            
        Returns:
            True if deleted successfully, False otherwise
        """
        try:
            self.collection.delete(ids=[job_id])
            logger.info(f"Deleted time series with ID: {job_id}")
            return True
        except Exception as e:
            logger.error(f"Error deleting time series: {str(e)}")
            return False
    
    def search_by_metadata(self, 
                         metadata_filter: Dict[str, Any],
                         limit: int = 10) -> List[Dict[str, Any]]:
        """
        Search for time series by metadata.
        
        Args:
            metadata_filter: Dictionary of metadata filters
            limit: Maximum number of results to return
            
        Returns:
            List of dictionaries containing matching time series info
        """
        try:
            # Prepare the filter
            where = {}
            for key, value in metadata_filter.items():
                if isinstance(value, (int, float)):
                    where[key] = {"$eq": value}
                elif isinstance(value, str):
                    where[key] = {"$eq": value}
                
            # Query the collection
            results = self.collection.get(
                where=where,
                limit=limit,
                include=["metadatas", "documents"]
            )
            
            # Format the results
            formatted_results = []
            if results and results["ids"]:
                for i, job_id in enumerate(results["ids"]):
                    formatted_results.append({
                        "job_id": job_id,
                        "metadata": results["metadatas"][i],
                        "description": results["documents"][i]
                    })
            
            return formatted_results
        except Exception as e:
            logger.error(f"Error searching by metadata: {str(e)}")
            return []