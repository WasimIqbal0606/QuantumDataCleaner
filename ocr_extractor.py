import pytesseract
import logging
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from typing import Dict, Any, List, Union, Tuple, Optional
import pdf2image
import io
import re
import os
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("logs/ocr_extractor.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("ocr_extractor")

# Make sure the logs directory exists
os.makedirs("logs", exist_ok=True)

class OCRExtractor:
    """
    Class implementing OCR extraction of time-series data from images and PDFs.
    Uses Tesseract OCR and image preprocessing to extract tabular data.
    Supports quantum-inspired noise reduction for better OCR quality.
    """
    
    def __init__(self):
        """Initialize the OCR extractor."""
        logger.info("Initializing OCRExtractor")
        
        # Check if tesseract is installed and available
        try:
            pytesseract.get_tesseract_version()
            self.tesseract_available = True
            logger.info("Tesseract OCR is available")
        except Exception as e:
            logger.warning(f"Tesseract OCR is not available: {str(e)}")
            logger.warning("Using fallback text extraction methods")
            self.tesseract_available = False
    
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess the image for better OCR.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Preprocessed image as numpy array
        """
        # Convert to grayscale if the image has multiple channels
        if len(image.shape) == 3 and image.shape[2] > 1:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
            
        # Apply thresholding to get a binary image
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Apply noise reduction
        denoised = cv2.fastNlMeansDenoising(binary, None, 10, 7, 21)
        
        return denoised
    
    def quantum_inspired_image_enhancement(self, image: np.ndarray) -> np.ndarray:
        """
        Apply quantum-inspired algorithms for image enhancement.
        This is a simulation of quantum image processing techniques.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Enhanced image as numpy array
        """
        # Convert to float for processing
        float_img = image.astype(np.float32) / 255.0
        
        # Apply quantum-inspired Fourier transform for noise reduction
        # Simulate quantum interference patterns
        rows, cols = float_img.shape
        crow, ccol = rows // 2, cols // 2
        
        # Perform FFT
        f = np.fft.fft2(float_img)
        fshift = np.fft.fftshift(f)
        
        # Create a mask (quantum-inspired bandpass filter)
        mask = np.zeros((rows, cols), np.uint8)
        center_radius = min(rows, cols) // 4
        outer_radius = min(rows, cols) // 2
        cv2.circle(mask, (ccol, crow), center_radius, 1, -1)
        cv2.circle(mask, (ccol, crow), outer_radius, 0, -1)
        
        # Apply mask
        fshift_filtered = fshift * mask
        
        # Inverse FFT
        f_ishift = np.fft.ifftshift(fshift_filtered)
        img_back = np.fft.ifft2(f_ishift)
        img_back = np.abs(img_back)
        
        # Normalize and convert back to uint8
        img_back = np.clip(img_back * 255.0, 0, 255).astype(np.uint8)
        
        # Apply contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(img_back)
        
        return enhanced
    
    def extract_text_from_image(self, image: Union[str, np.ndarray, Image.Image], 
                               use_quantum: bool = False) -> str:
        """
        Extract text from an image using OCR.
        
        Args:
            image: Input image (file path, numpy array, or PIL Image)
            use_quantum: Whether to use quantum-inspired image enhancement
            
        Returns:
            Extracted text as string
        """
        # Convert input to numpy array
        if isinstance(image, str):
            # Load image from file path
            img = cv2.imread(image)
        elif isinstance(image, Image.Image):
            # Convert PIL Image to numpy array
            img = np.array(image)
        else:
            # Assume it's already a numpy array
            img = image
            
        # Preprocess the image
        preprocessed = self.preprocess_image(img)
        
        # Apply quantum-inspired enhancement if requested
        if use_quantum:
            preprocessed = self.quantum_inspired_image_enhancement(preprocessed)
        
        # Extract text using Tesseract OCR
        if self.tesseract_available:
            try:
                text = pytesseract.image_to_string(preprocessed)
                return text
            except Exception as e:
                logger.error(f"Error extracting text with Tesseract: {str(e)}")
                return ""
        else:
            logger.warning("Tesseract not available, returning empty string")
            return ""
    
    def extract_tables_from_image(self, image: Union[str, np.ndarray, Image.Image], 
                                use_quantum: bool = False) -> List[pd.DataFrame]:
        """
        Extract tables from an image using OCR.
        
        Args:
            image: Input image (file path, numpy array, or PIL Image)
            use_quantum: Whether to use quantum-inspired image enhancement
            
        Returns:
            List of extracted tables as pandas DataFrames
        """
        # Convert input to numpy array
        if isinstance(image, str):
            # Load image from file path
            img = cv2.imread(image)
        elif isinstance(image, Image.Image):
            # Convert PIL Image to numpy array
            img = np.array(image)
        else:
            # Assume it's already a numpy array
            img = image
            
        # Preprocess the image
        preprocessed = self.preprocess_image(img)
        
        # Apply quantum-inspired enhancement if requested
        if use_quantum:
            preprocessed = self.quantum_inspired_image_enhancement(preprocessed)
        
        # Extract tables using Tesseract OCR
        if self.tesseract_available:
            try:
                # Use Tesseract's table extraction capability
                tables_data = pytesseract.image_to_data(preprocessed, output_type=pytesseract.Output.DATAFRAME)
                
                # Process the OCR data to extract tables
                tables = self._convert_ocr_data_to_tables(tables_data)
                return tables
            except Exception as e:
                logger.error(f"Error extracting tables with Tesseract: {str(e)}")
                return []
        else:
            logger.warning("Tesseract not available, returning empty list")
            return []
    
    def _convert_ocr_data_to_tables(self, ocr_data: pd.DataFrame) -> List[pd.DataFrame]:
        """
        Convert OCR data to tables.
        
        Args:
            ocr_data: OCR data as DataFrame
            
        Returns:
            List of extracted tables as pandas DataFrames
        """
        # Filter out non-text entries
        text_data = ocr_data[ocr_data['text'].notna() & (ocr_data['text'].str.strip() != '')]
        
        # Group by block_num to get individual tables
        tables = []
        for block_num, block_data in text_data.groupby('block_num'):
            # Sort by line and word position
            sorted_data = block_data.sort_values(by=['par_num', 'line_num', 'word_num'])
            
            # Group words into lines
            lines = []
            current_line = []
            current_line_num = -1
            
            for _, row in sorted_data.iterrows():
                if row['line_num'] != current_line_num:
                    if current_line:
                        lines.append(' '.join(current_line))
                    current_line = [row['text']]
                    current_line_num = row['line_num']
                else:
                    current_line.append(row['text'])
            
            # Add the last line
            if current_line:
                lines.append(' '.join(current_line))
            
            # Try to convert lines to a table
            if len(lines) > 1:
                # Split each line by whitespace to create rows
                rows = [re.split(r'\s{2,}', line) for line in lines]
                
                # Find the maximum number of columns
                max_cols = max(len(row) for row in rows)
                
                # Pad rows with fewer columns
                padded_rows = [row + [''] * (max_cols - len(row)) for row in rows]
                
                # Create a DataFrame
                if padded_rows:
                    table = pd.DataFrame(padded_rows[1:], columns=padded_rows[0])
                    tables.append(table)
        
        return tables
    
    def process_pdf(self, pdf_file: Union[str, bytes], 
                   use_quantum: bool = False) -> Tuple[str, List[pd.DataFrame]]:
        """
        Process a PDF file to extract text and tables.
        
        Args:
            pdf_file: Input PDF file (file path or bytes)
            use_quantum: Whether to use quantum-inspired image enhancement
            
        Returns:
            Tuple of (extracted_text, list_of_tables)
        """
        # Convert PDF to images
        if isinstance(pdf_file, str):
            # File path
            try:
                images = pdf2image.convert_from_path(pdf_file)
            except Exception as e:
                logger.error(f"Error converting PDF to images: {str(e)}")
                return "", []
        else:
            # Bytes
            try:
                images = pdf2image.convert_from_bytes(pdf_file)
            except Exception as e:
                logger.error(f"Error converting PDF bytes to images: {str(e)}")
                return "", []
        
        # Extract text and tables from each page
        full_text = ""
        all_tables = []
        
        for i, img in enumerate(images):
            logger.info(f"Processing PDF page {i+1}/{len(images)}")
            
            # Extract text
            text = self.extract_text_from_image(img, use_quantum=use_quantum)
            full_text += f"\n--- Page {i+1} ---\n{text}\n"
            
            # Extract tables
            tables = self.extract_tables_from_image(img, use_quantum=use_quantum)
            for j, table in enumerate(tables):
                table.name = f"Page_{i+1}_Table_{j+1}"
                all_tables.append(table)
        
        return full_text, all_tables
    
    def extract_time_series(self, 
                           source: Union[str, bytes, np.ndarray, Image.Image],
                           source_type: str,
                           date_column: Optional[str] = None,
                           value_column: Optional[str] = None,
                           use_quantum: bool = False) -> Tuple[Optional[pd.DataFrame], Dict[str, Any]]:
        """
        Extract time-series data from various sources.
        
        Args:
            source: Input source (file path, bytes, numpy array, or PIL Image)
            source_type: Type of the source ('image', 'pdf', 'text')
            date_column: Name of the date column (optional)
            value_column: Name of the value column (optional)
            use_quantum: Whether to use quantum-inspired enhancement
            
        Returns:
            Tuple of (extracted_dataframe, metadata)
        """
        start_time = time.time()
        logger.info(f"Extracting time-series data from {source_type}")
        
        extracted_df = None
        metadata = {
            "source_type": source_type,
            "use_quantum": use_quantum,
            "extraction_successful": False,
            "tables_found": 0,
            "time_series_found": False,
            "date_column": None,
            "value_column": None
        }
        
        try:
            # Extract tables based on source type
            tables = []
            
            if source_type == 'image':
                tables = self.extract_tables_from_image(source, use_quantum=use_quantum)
            elif source_type == 'pdf':
                _, tables = self.process_pdf(source, use_quantum=use_quantum)
            elif source_type == 'text':
                # For text sources, assume it's already a CSV-like format
                if isinstance(source, str):
                    if os.path.isfile(source):
                        # It's a file path
                        with open(source, 'r') as f:
                            content = f.read()
                    else:
                        # It's raw text
                        content = source
                else:
                    # It's bytes
                    content = source.decode('utf-8')
                
                # Try to parse as CSV
                try:
                    from io import StringIO
                    extracted_df = pd.read_csv(StringIO(content))
                    tables = [extracted_df]
                except Exception as e:
                    logger.error(f"Error parsing text as CSV: {str(e)}")
            
            metadata["tables_found"] = len(tables)
            
            # Find time-series data in the tables
            if tables:
                # Look for the most promising table
                best_table = None
                best_date_col = None
                best_value_col = None
                best_score = 0
                
                for table in tables:
                    score, date_col, value_col = self._evaluate_table_for_time_series(table, date_column, value_column)
                    if score > best_score:
                        best_score = score
                        best_table = table
                        best_date_col = date_col
                        best_value_col = value_col
                
                if best_table is not None and best_date_col is not None:
                    extracted_df = best_table
                    metadata["time_series_found"] = True
                    metadata["date_column"] = best_date_col
                    metadata["value_column"] = best_value_col
                    metadata["extraction_successful"] = True
                    
                    # Try to convert date column to datetime
                    try:
                        extracted_df[best_date_col] = pd.to_datetime(extracted_df[best_date_col])
                    except Exception as e:
                        logger.warning(f"Could not convert date column to datetime: {str(e)}")
                    
                    # Sort by date
                    try:
                        extracted_df = extracted_df.sort_values(by=best_date_col)
                    except Exception as e:
                        logger.warning(f"Could not sort by date column: {str(e)}")
            
        except Exception as e:
            logger.error(f"Error extracting time-series data: {str(e)}")
        
        # Calculate execution time
        execution_time_ms = (time.time() - start_time) * 1000
        logger.info(f"Time-series extraction completed in {execution_time_ms:.2f} ms")
        metadata["execution_time_ms"] = execution_time_ms
        
        return extracted_df, metadata
    
    def _evaluate_table_for_time_series(self, 
                                       table: pd.DataFrame,
                                       preferred_date_column: Optional[str] = None,
                                       preferred_value_column: Optional[str] = None) -> Tuple[float, Optional[str], Optional[str]]:
        """
        Evaluate a table for time-series data.
        
        Args:
            table: Input table as DataFrame
            preferred_date_column: Preferred date column name (optional)
            preferred_value_column: Preferred value column name (optional)
            
        Returns:
            Tuple of (score, date_column, value_column)
        """
        score = 0
        date_column = None
        value_column = None
        
        # Check if the table has at least 2 columns
        if len(table.columns) < 2:
            return 0, None, None
        
        # Try to find date column
        date_col_candidates = []
        
        # Check if the preferred date column exists
        if preferred_date_column and preferred_date_column in table.columns:
            date_column = preferred_date_column
            date_col_candidates = [preferred_date_column]
        else:
            # Look for columns that could be dates
            for col in table.columns:
                col_name = str(col).lower()
                if ('date' in col_name or 'time' in col_name or 'day' in col_name or 
                    'month' in col_name or 'year' in col_name):
                    date_col_candidates.append(col)
                    score += 10  # Bonus for having a column with date-like name
            
            # If no date-like column names, try to identify by content
            if not date_col_candidates:
                for col in table.columns:
                    # Check if the column can be converted to datetime
                    try:
                        pd.to_datetime(table[col], errors='raise')
                        date_col_candidates.append(col)
                        score += 5  # Bonus for having a column that can be converted to datetime
                    except:
                        pass
            
            # Select the first candidate as the date column
            if date_col_candidates:
                date_column = date_col_candidates[0]
        
        # If we found a date column, try to find a value column
        if date_column:
            value_col_candidates = []
            
            # Check if the preferred value column exists
            if preferred_value_column and preferred_value_column in table.columns:
                value_column = preferred_value_column
                value_col_candidates = [preferred_value_column]
            else:
                # Look for columns that could be values
                for col in table.columns:
                    if col == date_column:
                        continue
                        
                    col_name = str(col).lower()
                    # Check for value-like names
                    if ('value' in col_name or 'price' in col_name or 'amount' in col_name or 
                        'qty' in col_name or 'quantity' in col_name or 'rate' in col_name or 
                        'ratio' in col_name or 'percent' in col_name):
                        value_col_candidates.append(col)
                        score += 10  # Bonus for having a column with value-like name
                
                # If no value-like column names, look for numeric columns
                if not value_col_candidates:
                    for col in table.columns:
                        if col == date_column:
                            continue
                            
                        # Check if the column is numeric
                        if pd.api.types.is_numeric_dtype(table[col]):
                            value_col_candidates.append(col)
                            score += 5  # Bonus for having a numeric column
                
                # Select the first candidate as the value column
                if value_col_candidates:
                    value_column = value_col_candidates[0]
                else:
                    # If still no candidate, use the first non-date column
                    for col in table.columns:
                        if col != date_column:
                            value_column = col
                            break
        
        # Give a higher score to tables with more rows (likely time series)
        score += min(len(table), 100) / 10  # Up to 10 points for up to 100 rows
        
        return score, date_column, value_column