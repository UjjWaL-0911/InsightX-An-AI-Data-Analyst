import pandas as pd
import numpy as np
from typing import Dict, List, Any

class DataPreprocessor:
    def __init__(self):
        self.numeric_columns = []
        self.categorical_columns = []
        self.datetime_columns = []
    
    def preprocess_dataframe(self, df: pd.DataFrame) -> tuple[pd.DataFrame, str]:
        """Preprocess dataframe data and return the processed dataframe along with a message."""
        message = "Data preprocessing completed: Missing values handled, types converted, duplicates removed."
        
        # Create a copy to avoid modifying original
        df_cleaned = df.copy()
        
        # Handle missing values
        df_cleaned = self._handle_missing_values(df_cleaned)
        
        # Convert data types
        df_cleaned = self._convert_data_types(df_cleaned)
        
        # Remove duplicates
        original_rows = len(df_cleaned)
        df_cleaned = df_cleaned.drop_duplicates()
        if len(df_cleaned) < original_rows:
            message += f" Removed {original_rows - len(df_cleaned)} duplicate rows."
        
        return df_cleaned, message
    
    def preprocess_text(self, text: str) -> str:
        """Preprocess text data"""
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        # Remove special characters but keep basic punctuation
        text = ''.join(char for char in text if char.isalnum() or char.isspace() or char in '.,!?')
        
        return text.strip()
    
    def analyze_structure(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze data structure"""
        # Reset column lists
        self.numeric_columns = []
        self.categorical_columns = []
        self.datetime_columns = []
        
        # Analyze each column
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                self.numeric_columns.append(col)
            elif pd.api.types.is_datetime64_dtype(df[col]):
                self.datetime_columns.append(col)
            else:
                self.categorical_columns.append(col)
        
        return {
            'numeric_columns': self.numeric_columns,
            'categorical_columns': self.categorical_columns,
            'datetime_columns': self.datetime_columns,
            'total_columns': len(df.columns),
            'total_rows': len(df)
        }
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in dataframe"""
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                # Fill numeric columns with median
                df[col] = df[col].fillna(df[col].median() if not df[col].empty else 0) # Added check for empty series
            elif pd.api.types.is_object_dtype(df[col]): # Handle object type specifically for mode
                # Fill non-numeric/object columns with mode, if mode exists
                # Ensure the mode is not empty, if it is, fill with an empty string
                mode_val = df[col].mode()
                if not mode_val.empty:
                    df[col] = df[col].fillna(mode_val[0])
                else:
                    df[col] = df[col].fillna('') # Fallback for columns with no mode (e.g., all unique NaNs)
            else:
                # For other types (e.g., datetime), forward fill or drop, or fill with None/NaT
                # For simplicity, filling with NaN for now for non-numeric/object types where mode might not apply well
                df[col] = df[col].fillna(np.nan) # Using np.nan for other dtypes where applicable
        return df
    
    def _convert_data_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert data types to appropriate formats for Arrow compatibility."""
        for col in df.columns:
            # Step 1: Ensure Python None becomes numpy NaN for consistent handling across types
            if df[col].isnull().any(): # Only apply if there are nulls to optimize
                df[col] = df[col].replace({None: np.nan})

            # Step 2: Skip if already in suitable Streamlit/Arrow compatible types
            if pd.api.types.is_datetime64_dtype(df[col]) or \
               pd.api.types.is_string_dtype(df[col]) or \
               pd.api.types.is_categorical_dtype(df[col]):
                continue
            
            # Step 3: Attempt to convert to datetime
            date_keywords = ['date', 'time', 'year', 'month', 'day', 'hour', 'minute', 'second']
            if any(keyword in col.lower() for keyword in date_keywords):
                converted_col = pd.to_datetime(df[col], errors='coerce')
                if not converted_col.isna().all(): # If at least some values converted
                    df[col] = converted_col
                    continue
            
            # Step 4: Attempt to convert to numeric (handling nullable integers and floats)
            # Try converting to numeric, coercing errors to NaN
            converted_col = pd.to_numeric(df[col], errors='coerce')
            
            # Check if conversion was successful for more than 50% of non-NaN values
            if not converted_col.isna().all() and (converted_col.notna().sum() / len(converted_col) > 0.5):
                # If it can be a nullable integer (all non-NaN values are integers),
                # convert to pd.Int64Dtype() for Arrow compatibility with NaNs
                if (converted_col.dropna() % 1 == 0).all():
                    df[col] = converted_col.astype(pd.Int64Dtype())
                else:
                    # Otherwise, convert to float64 for general numeric compatibility with NaNs
                    df[col] = converted_col.astype(np.float64)
                continue # Move to next column if successfully converted to numeric
            
            # Step 5: If still an object type after numeric/datetime attempts, convert to nullable string
            if df[col].dtype == 'object':
                # First, convert all values to their string representation to handle mixed types robustly
                df[col] = df[col].astype(str)
                # Then, convert to nullable StringDtype for Arrow compatibility and better memory usage
                df[col] = df[col].astype(pd.StringDtype())
                # Finally, fill any remaining NaNs (which might be 'nan' strings after astype(str)) with an empty string
                df[col] = df[col].fillna('')
            
            # Step 6: If column has few unique values, convert to category
            # This should be done after string conversion to ensure consistency
            # Also ensure there are actual non-null values before converting to category
            if df[col].nunique() < len(df) * 0.1 and len(df[col].dropna()) > 0:
                df[col] = df[col].astype('category')
        
        return df


