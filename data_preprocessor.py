import pandas as pd
import numpy as np
from typing import Dict, Any

class DataPreprocessor:
    def __init__(self, agent=None):
        self.agent = agent

    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in dataframe"""
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                # Fill numeric columns with median
                df[col] = df[col].fillna(df[col].median() if not df[col].empty else 0)
            elif pd.api.types.is_object_dtype(df[col]):
                # Fill text columns with mode or empty string
                df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else '')
            else:
                # Fill other types with NaN
                df[col] = df[col].fillna(np.nan)
        return df
    
    def _convert_data_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert data types based on LLM's analysis of the dataset."""
        if not self.agent:
            raise ValueError("Agent is required for type conversion. Please initialize DataPreprocessor with an agent.")

        try:
            # Prepare dataset info for LLM
            dataset_info = {
                'columns': list(df.columns),
                'sample_data': df.head().to_dict(),
                'dtypes': df.dtypes.astype(str).to_dict(),
                'null_counts': df.isnull().sum().to_dict(),
                'unique_counts': {col: df[col].nunique() for col in df.columns}
            }
            
            # Get LLM's type recommendations
            type_recommendations = self.agent.get_column_types(dataset_info)
            
            # Convert columns according to LLM's recommendations
            for col, target_type in type_recommendations.items():
                if col in df.columns:
                    try:
                        # Convert to the recommended type
                        df[col] = df[col].astype(target_type)
                        print(f"Converted {col} to {target_type}")
                    except Exception as e:
                        print(f"Warning: Could not convert {col} to {target_type}: {str(e)}")
                        # Fallback to string type if conversion fails
                        df[col] = df[col].astype('string')
            
            return df
            
        except Exception as e:
            raise Exception(f"Error in type conversion: {str(e)}")

    def preprocess_dataframe(self, df: pd.DataFrame) -> tuple[pd.DataFrame, str]:
        """Preprocess dataframe data and return the processed dataframe along with a message."""
        message = "Data preprocessing completed: Missing values handled, types converted, duplicates removed."
        
        # Create a copy to avoid modifying original
        df_cleaned = df.copy()
        df_cleaned = self._convert_data_types(df_cleaned)
        df_cleaned = self._handle_missing_values(df_cleaned)
        
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
        text = ''.join(char for char in text if char.isalnum() or char.isspace() or char in '.,!?;:')
        
        return text.strip()
    