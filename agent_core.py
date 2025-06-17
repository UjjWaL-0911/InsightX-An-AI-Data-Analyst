#Intelligent Agent Core

import json
import pickle
import uuid
from datetime import datetime
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import pandas as pd
import re
from collections import deque
import together
import os
from together import Together
import time
import matplotlib.pyplot as plt
import io
import base64
import numpy as np
from dotenv import load_dotenv
from file_processor import FileProcessor

# Create required directories
for dir_name in ["Graph_Plots", "Saved_Sessions", "Datasets", "Configs"]:
    os.makedirs(dir_name, exist_ok=True)

# Load environment variables
env_path = os.path.join("Configs", ".env")
if os.path.exists(env_path):
    load_dotenv(env_path)

#Context Management System
class ContextManager:
    """Manages context and state for the agent"""
    
    @dataclass
    class DatasetState:
        filename: str
        columns: List[str]
        shape: Tuple[int, int]
        data_types: Dict[str, str]
        loaded_at: datetime
    
    def __init__(self):
        self.session_id = str(uuid.uuid4())
        self.current_dataset = None
        self.current_data = None
        self.conversation_history = []
        self.dataset_states = {}
    
    def register_dataset(self, filepath: str, file_info: Dict[str, Any]) -> str:
        """Register a new dataset"""
        self.current_dataset = filepath
        self.current_data = file_info['data']
        
        # Store dataset state
        self.dataset_states[filepath] = self.DatasetState(
            filename=filepath,
            columns=list(file_info['data'].columns),
            shape=file_info['data'].shape,
            data_types=file_info['data'].dtypes.astype(str).to_dict(),
            loaded_at=datetime.now()
        )
        
        return f"Dataset registered: {filepath}"
    
    def add_conversation(self, user_query: str, agent_response: str):
        """Add a conversation turn to history"""
        self.conversation_history.append({
            'timestamp': datetime.now().isoformat(),
            'user_query': user_query,
            'agent_response': agent_response
        })
    
    def get_current_context(self) -> Dict[str, Any]:
        """Get current context information"""
        if not self.current_dataset:
            return {}
        
        return {
            'dataset': self.current_dataset,
            'state': self.dataset_states.get(self.current_dataset),
            'conversation_count': len(self.conversation_history)
        }
    
    def save_session(self, filepath: str = None):
        """Save current session state"""
        if not filepath:
            filepath = f"session_{self.session_id}.json"
        
        session_data = {
            'session_id': self.session_id,
            'current_dataset': self.current_dataset,
            'conversation_history': self.conversation_history,
            'dataset_states': {
                k: {
                    'filename': v.filename,
                    'columns': v.columns,
                    'shape': v.shape,
                    'data_types': v.data_types,
                    'loaded_at': v.loaded_at.isoformat()
                }
                for k, v in self.dataset_states.items()
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(session_data, f, indent=2)
    
    def load_session(self, filepath: str = None):
        """Load session state from file"""
        if not filepath:
            return
        
        with open(filepath, 'r') as f:
            session_data = json.load(f)
        
        self.session_id = session_data['session_id']
        self.current_dataset = session_data['current_dataset']
        self.conversation_history = session_data['conversation_history']
        
        # Reconstruct dataset states
        self.dataset_states = {
            k: self.DatasetState(
                filename=v['filename'],
                columns=v['columns'],
                shape=tuple(v['shape']),
                data_types=v['data_types'],
                loaded_at=datetime.fromisoformat(v['loaded_at'])
            )
            for k, v in session_data['dataset_states'].items()
        }


#LLM Integration
class LLMInterface:
    """Interface for interacting with the LLM"""
    
    def __init__(self, api_key: str, context_manager: ContextManager):
        self.api_key = api_key
        self.context_manager = context_manager
        self.client = Together(api_key=api_key)
    
    def generate_response(self, prompt: str, system_prompt: str = None) -> str:
        """Generate response from LLM"""
        if not system_prompt:
            system_prompt = "You are an intelligent data analyst agent. Provide direct, actionable analysis and answers about the data."
        
        messages = [{"role": "system", "content": system_prompt}]
        messages.append({"role": "user", "content": prompt})
        
        response = self.client.chat.completions.create(
            model="mistralai/Mixtral-8x7B-Instruct-v0.1",
            messages=messages,
            max_tokens=1024,
            temperature=0.7,
            top_p=0.7,
            top_k=50,
            repetition_penalty=1.1,
            stop=['</s>', 'Human:', 'Assistant:']
        )
        
        return response.choices[0].message.content.strip()
    
    def generate_visualization(self, data_description: str, visualization_type: str) -> str:
        """Generate visualization based on data description"""
        # Create a temporary file for the plot
        plot_path = f"temp_plot_{uuid.uuid4()}.png"
        
        try:
            # Generate Python code for visualization
            code_prompt = f"""Generate Python code to create a {visualization_type} based on this description:
            {data_description}
            
            The code should:
            1. Use matplotlib or seaborn
            2. Save the plot to '{plot_path}'
            3. Be complete and executable
            4. Handle any necessary data transformations
            
            Return only the Python code, no explanations."""
            
            code = self.generate_response(code_prompt, "You are a data visualization expert. Generate clean, efficient visualization code.")
            
            # Execute the code
            exec(code)
            
            # Convert plot to base64
            with open(plot_path, 'rb') as f:
                base64_data = base64.b64encode(f.read()).decode()
            
            return f"data:image/png;base64,{base64_data}"
            
        except Exception as e:
            raise Exception(f"Error generating visualization: {str(e)}")
        
        finally:
            # Clean up
            if os.path.exists(plot_path):
                os.remove(plot_path)


# Main Intelligent Agent Class
class IntelligentAgent:
    """Main agent that combines all components"""
    
    def __init__(self, together_api_key: str):
        self.together_api_key = together_api_key
        self.context_manager = ContextManager()
        self.llm_interface = LLMInterface(together_api_key, self.context_manager)
        self.file_processor = FileProcessor()
    
    def register_dataset(self, filepath: str, file_type: str) -> str:
        """Register new dataset with agent"""
        # Create a new session
        self.context_manager = ContextManager()
        self.llm_interface = LLMInterface(self.together_api_key, self.context_manager)
        
        # Process and register the file
        file_info = self.file_processor.process_file(filepath)
        return self.context_manager.register_dataset(filepath, file_info)
    
    def process_query(self, user_query: str) -> Dict[str, Any]:
        """Process user query and generate response"""
        current_data = self.context_manager.current_data
        
        # Generate context-aware prompt
        context_prompt = f"User Query: {user_query}\n\n"
        
        # Add data context
        if isinstance(current_data, pd.DataFrame):
            context_prompt += f"Complete Dataset Information:\n"
            context_prompt += f"Total Rows: {len(current_data)}\n"
            context_prompt += f"Columns: {', '.join(current_data.columns)}\n"
            context_prompt += f"Data Types:\n{current_data.dtypes.to_string()}\n"
            context_prompt += f"\nSummary Statistics:\n{current_data.describe().to_string()}\n"
            
            # Add specific analysis for price/area queries
            if 'price' in current_data.columns and 'area' in current_data.columns:
                price_area_ratio = current_data['price'] / current_data['area']
                min_ratio_idx = price_area_ratio.idxmin()
                best_house = current_data.loc[min_ratio_idx]
                
                context_prompt += f"\nPrice/Area Analysis:\n"
                context_prompt += f"Minimum Price/Area Ratio: {price_area_ratio.min():.2f}\n"
                context_prompt += f"Maximum Price/Area Ratio: {price_area_ratio.max():.2f}\n"
                context_prompt += f"Average Price/Area Ratio: {price_area_ratio.mean():.2f}\n"
                context_prompt += f"\nHouse with Best Price/Area Ratio:\n{best_house.to_string()}\n"
        else:
            context_prompt += f"\nContent:\n{current_data}\n"
        
        # Generate response
        system_prompt = "You are an intelligent data analyst agent. Provide direct, actionable analysis and answers about the data."
        llm_response = self.llm_interface.generate_response(context_prompt, system_prompt)
        
        # Handle visualization if needed
        visualization = None
        if isinstance(current_data, pd.DataFrame) and any(word in user_query.lower() for word in ['plot', 'chart', 'graph', 'visualize', 'show']):
            data_description = llm_response.split("Visualization:")[-1].strip()
            visualization = self.llm_interface.generate_visualization(data_description, "visualization")
        
        # Store conversation
        self.context_manager.add_conversation(user_query, llm_response)
        
        return {
            'user_query': user_query,
            'llm_response': llm_response,
            'visualization': visualization
        }
    
    def get_conversation_history(self) -> List[Dict]:
        """Get formatted conversation history"""
        return list(self.context_manager.conversation_history)
    
    def save_session(self):
        """Save current session state"""
        self.context_manager.save_session()
    
    def get_agent_status(self) -> Dict[str, Any]:
        """Get current agent status"""
        return {
            'session_id': self.context_manager.session_id,
            'current_dataset': self.context_manager.current_dataset,
            'conversation_count': len(self.context_manager.conversation_history)
        }

    def get_column_types(self, dataset_info: dict) -> dict:
        """
        Analyze dataset and recommend appropriate data types for each column using LLM.
        
        Args:
            dataset_info (dict): Dictionary containing:
                - columns: List of column names
                - sample_data: Dictionary of sample values for each column
                - dtypes: Current data types of columns
                - null_counts: Number of null values in each column
                - unique_counts: Number of unique values in each column
        
        Returns:
            dict: Mapping of column names to their recommended numpy/pandas data types
        """
        # Create a detailed prompt for the LLM
        prompt = f"""You are a data type expert. Your task is to analyze the dataset information and recommend the most appropriate data type for each column.
You must respond with ONLY a valid JSON object, nothing else.

Consider these factors for each column:
1. The nature of the data (numeric, text, date, etc.)
2. The number of unique values (for categorical vs string decisions)
3. The presence of null values
4. The current data type
5. The sample values

Dataset Information:
Columns: {dataset_info['columns']}
Current Data Types: {dataset_info['dtypes']}
Null Counts: {dataset_info['null_counts']}
Unique Value Counts: {dataset_info['unique_counts']}
Sample Data: {dataset_info['sample_data']}

For each column, you must choose EXACTLY one of these data types:
- int64: For any column containing only integers, including:
  * Integer numbers
  * Numeric categorical data (IDs, codes)
  * Float numbers that are actually integers (e.g., 1.0, 2.0)
  * Any column where all non-null values are integers
- float64: For decimal numbers that are not integers
- category: For non-numeric categorical data with few unique values (e.g., status, type, or any text-based categories)
- datetime64[ns]: For date/time values
- string: For text data with many unique values

Important Rules:
1. If a column contains only integers (even if it's categorical or currently float), use int64
2. Only use category for non-numeric categorical data
3. Use string for text data with many unique values
4. Use float64 only for decimal numbers that are not integers
5. Use datetime64[ns] for date/time values
6. Check if float values are actually integers (e.g., 1.0, 2.0) and use int64 in such cases

Your response must be a valid JSON object where:
- Keys are the column names
- Values are the recommended data types
- No additional text or explanation
- No markdown formatting
- No code blocks

Example of expected response format:
{{
    "column1": "int64",
    "column2": "float64",
    "column3": "category"
}}

Remember: Return ONLY the JSON object, nothing else.No Ok,understood,nothing else"""

        # Get LLM's response
        response = self.llm_interface.generate_response(prompt)
        
        try:
            # Clean the response to ensure it's valid JSON
            # Remove any markdown formatting or extra text
            response = response.strip()
            if response.startswith('```json'):
                response = response[7:]
            if response.endswith('```'):
                response = response[:-3]
            response = response.strip()
            
            # Parse the response as JSON
            type_recommendations = json.loads(response)
            
            # Validate the recommendations
            valid_types = {'int64', 'float64', 'category', 'datetime64[ns]', 'string'}
            for col, dtype in type_recommendations.items():
                if dtype not in valid_types:
                    raise ValueError(f"Invalid data type recommendation: {dtype}")
                if col not in dataset_info['columns']:
                    raise ValueError(f"Column not found in dataset: {col}")
            
            return type_recommendations
            
        except json.JSONDecodeError as e:
            print(f"Raw LLM response: {response}")  # Debug print
            raise ValueError(f"Failed to parse LLM response as JSON: {str(e)}")
        except Exception as e:
            raise ValueError(f"Error processing type recommendations: {str(e)}")


