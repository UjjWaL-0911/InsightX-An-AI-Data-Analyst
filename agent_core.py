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
    """Manages conversation history and dataset state"""

    @dataclass
    class DatasetState:
        filename: str
        columns: List[str]
        shape: Tuple[int, int]
        data_types: Dict[str, str]
        loaded_at: datetime
        
    def __init__(self):
        self.conversation_history = deque(maxlen=50)
        self.current_dataset = None
        self.current_data = None
        self.dataset_registry = {}
        self.session_id = str(uuid.uuid4())[:8]
    
    def register_dataset(self, filepath: str, file_info: Dict[str, Any]) -> str:
        """Register new dataset in agent memory"""
        filename = os.path.basename(filepath)
        
        # Create dataset state
        dataset_state = self.DatasetState(
            filename=filename,
            columns=file_info['columns'] if file_info['type'] == 'dataframe' else ['content'],
            shape=file_info['shape'] if file_info['type'] == 'dataframe' else (1, 1),
            data_types={col: str(file_info['data'][col].dtype) for col in file_info['columns']} if file_info['type'] == 'dataframe' else {'content': 'text'},
            loaded_at=datetime.now()
        )
        
        # Store dataset state and data
        self.dataset_registry[filename] = dataset_state
        self.current_dataset = filepath
        self.current_data = file_info['data']
        
        return filename
    
    def add_conversation(self, user_query: str, agent_response: str, intent: str):
        """Add conversation to memory"""
        self.conversation_history.append({
            'timestamp': datetime.now(),
            'user_query': user_query,
            'agent_response': agent_response,
            'intent': intent,
            'dataset_context': self.current_dataset or "None"
        })
    
    def get_current_context(self) -> Dict[str, Any]:
        """Get current dataset and conversation context"""
        context = {
            'session_id': self.session_id,
            'current_dataset': self.current_dataset,
            'conversation_count': len(self.conversation_history)
        }
        
        if self.current_dataset in self.dataset_registry:
            dataset = self.dataset_registry[self.current_dataset]
            context['dataset_info'] = {
                'filename': dataset.filename,
                'shape': dataset.shape,
                'columns': dataset.columns
            }
        
        context['recent_conversations'] = list(self.conversation_history)[-3:]
        return context
    
    def save_session(self, filepath: str = None):
        """Save session state with unique filename"""
        if filepath is None:
            session_num = 1
            while os.path.exists(os.path.join("Saved_Sessions", f"session_state{session_num}.pkl")):
                session_num += 1
            filepath = os.path.join("Saved_Sessions", f"session_state{session_num}.pkl")
        
        with open(filepath, 'wb') as f:
            pickle.dump({
                'conversation_history': list(self.conversation_history),
                'dataset_registry': self.dataset_registry,
                'current_dataset': self.current_dataset,
                'session_id': self.session_id
            }, f)
        print(f"Session saved as: {filepath}")
    
    def load_session(self, filepath: str = None):
        """Load session state"""
        if filepath is None:
            session_files = [f for f in os.listdir("Saved_Sessions") if f.startswith("session_state") and f.endswith(".pkl")]
            if not session_files:
                print("No previous session found. Starting fresh.")
                return
            session_files.sort(key=lambda x: int(''.join(filter(str.isdigit, x))))
            filepath = os.path.join("Saved_Sessions", session_files[-1])
        
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.conversation_history = deque(data.get('conversation_history', []), maxlen=50)
            self.dataset_registry = data.get('dataset_registry', {})
            self.current_dataset = data.get('current_dataset')
            self.session_id = data.get('session_id', self.session_id)
        print(f"Loaded session from: {filepath}")


#Intent Classification Engine
class IntentType(Enum):
    """Possible user intents"""
    DATA_ANALYSIS = "data_analysis"
    VISUALIZATION = "visualization"
    QUESTION_ANSWER = "question_answer"
    COMPARISON = "comparison"
    FILTERING = "filtering"
    CORRELATION = "correlation"
    TREND_ANALYSIS = "trend_analysis"
    SUMMARY = "summary"
    OUTLIER_DETECTION = "outlier_detection"
    EXPORT = "export"
    HELP = "help"
    FILE_UPLOAD = "file_upload"
    UNCLEAR = "unclear"

class IntentClassifier:
    """Intent classification and query preprocessing"""
    
    def __init__(self, context_manager: ContextManager):
        self.context_manager = context_manager
        self.intent_patterns = {
            IntentType.DATA_ANALYSIS: [r'\b(analyz|stats|mean|insights)\b'],
            IntentType.VISUALIZATION: [r'\b(plot|chart|graph|visualiz)\b'],
            IntentType.QUESTION_ANSWER: [r'\b(what|how|why|when|where)\b', r'\?'],
            IntentType.COMPARISON: [r'\b(compar|vs|differ|between)\b'],
            IntentType.FILTERING: [r'\b(filter|where|select|subset)\b'],
            IntentType.CORRELATION: [r'\b(correlat|relationship|related)\b'],
            IntentType.TREND_ANALYSIS: [r'\b(trend|pattern|over time)\b'],
            IntentType.SUMMARY: [r'\b(summary|overview|brief)\b'],
            IntentType.OUTLIER_DETECTION: [r'\b(outlier|anomal|unusual)\b'],
            IntentType.EXPORT: [r'\b(export|download|save)\b'],
            IntentType.HELP: [r'\b(help|assist|guide)\b']
        }
    
    def classify_intent(self, query: str) -> Tuple[IntentType, float]:
        """Classify user intent with confidence score"""
        query_lower = query.lower()
        intent_scores = {}
        
        for intent, patterns in self.intent_patterns.items():
            score = sum(len(re.findall(pattern, query_lower)) for pattern in patterns)
            if score > 0:
                intent_scores[intent] = score / len(query.split())
        
        if intent_scores:
            best_intent = max(intent_scores.items(), key=lambda x: x[1])
            return best_intent[0], min(best_intent[1] * 2, 1.0)
        
        return IntentType.UNCLEAR, 0.0
    
    def generate_context_prompt(self, query: str, intent: IntentType) -> str:
        """Create contextual prompt for LLM"""
        context = self.context_manager.get_current_context()
        prompt = f"User Query: {query}\nDetected Intent: {intent.value}\n\n"
        
        if 'dataset_info' in context:
            dataset = context['dataset_info']
            prompt += f"Dataset: {dataset['filename']}\n"
            prompt += f"Shape: {dataset['shape']}\n"
            prompt += f"Columns: {', '.join(dataset['columns'])}\n"
        
        if context['recent_conversations']:
            prompt += "\nRecent Conversations:\n"
            for conv in context['recent_conversations']:
                prompt += f"- {conv['user_query']} ({conv['intent']})\n"
        
        return prompt


#LLM Integration
class LLMInterface:
    """Interface to Together.ai Llama model"""
    
    def __init__(self, api_key: str, context_manager: ContextManager):
        self.client = Together(api_key=api_key)
        self.model_name = "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8"
        self.max_tokens = 2048
        self.temperature = 0.7
        self.max_retries = 5
        self.initial_retry_delay = 1
        self.context_manager = context_manager
        
    def generate_response(self, prompt: str, system_prompt: str = None) -> str:
        """Generate response using LLM with retry logic"""
        for attempt in range(self.max_retries):
            try:
                messages = [{"role": "system", "content": system_prompt}] if system_prompt else []
                messages.append({"role": "user", "content": prompt})
                
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature
                )
                return response.choices[0].message.content.strip()
                
            except together.error.RateLimitError as e:
                if attempt < self.max_retries - 1:
                    wait_time = self.initial_retry_delay * (2 ** attempt)
                    print(f"Rate limit hit. Retrying in {wait_time:.2f} seconds...")
                    time.sleep(wait_time)
                    continue
                return f"Error: {str(e)}"
            except Exception as e:
                return f"Error: {str(e)}"
    
    def generate_visualization(self, data_description: str, visualization_type: str) -> str:
        """Generate and execute plotting code"""
        if not self.context_manager.current_dataset:
            return None
        
        df = self.context_manager.current_data
        prompt = f"""Given this data description: {data_description}
        Generate Python code using matplotlib to create an appropriate visualization.
        The code should:
        1. Use the DataFrame 'df' that is already loaded
        2. Create a clear and professional visualization
        3. Include proper labels, title, and styling
        4. Save the plot as a PNG file in the Graph_Plots folder
        
        IMPORTANT: The DataFrame columns are: {', '.join(df.columns)}
        Use these EXACT column names in your code.
        
        Return ONLY the Python code, nothing else."""

        plotting_code = self.generate_response(prompt)
        
        # Extract code from response
        if "```python" in plotting_code:
            plotting_code = plotting_code.split("```python")[1].split("```")[0].strip()
        elif "```" in plotting_code:
            plotting_code = plotting_code.split("```")[1].strip()
        
        # Execute code
        namespace = {'df': df, 'plt': plt, 'np': np, 'pd': pd}
        exec(plotting_code, namespace)
        
        # Save the plot
        plot_path = os.path.join("Graph_Plots", "temp_plot.png")
        plt.savefig(plot_path)
        plt.close()
        
        # Convert to base64
        with open(plot_path, "rb") as img_file:
            base64_data = base64.b64encode(img_file.read()).decode('utf-8')
        
        # Clean up
        if os.path.exists(plot_path):
            os.remove(plot_path)
        
        return f"data:image/png;base64,{base64_data}"


# Main Intelligent Agent Class
class IntelligentAgent:
    """Main agent that combines all components"""
    
    def __init__(self, together_api_key: str):
        self.together_api_key = together_api_key
        self.context_manager = ContextManager()
        self.intent_classifier = IntentClassifier(self.context_manager)
        self.llm_interface = LLMInterface(together_api_key, self.context_manager)
        self.file_processor = FileProcessor()
    
    def register_dataset(self, filepath: str, file_type: str) -> str:
        """Register new dataset with agent"""
        # Create a new session
        self.context_manager = ContextManager()
        self.intent_classifier = IntentClassifier(self.context_manager)
        self.llm_interface = LLMInterface(self.together_api_key, self.context_manager)
        
        # Process and register the file
        file_info = self.file_processor.process_file(filepath)
        return self.context_manager.register_dataset(filepath, file_info)
    
    def process_query(self, user_query: str) -> Dict[str, Any]:
        """Process user query and generate response"""
        # Classify intent and get response
        intent, confidence = self.intent_classifier.classify_intent(user_query)
        current_data = self.context_manager.current_data
        
        # Generate context-aware prompt
        context_prompt = self.intent_classifier.generate_context_prompt(user_query, intent)
        
        # Add data context
        if isinstance(current_data, pd.DataFrame):
            context_prompt += f"\nComplete Dataset Information:\n"
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
        if intent == IntentType.VISUALIZATION and isinstance(current_data, pd.DataFrame):
            data_description = llm_response.split("Visualization:")[-1].strip()
            visualization = self.llm_interface.generate_visualization(data_description, "visualization")
        
        # Store conversation
        self.context_manager.add_conversation(user_query, llm_response, intent.value)
        
        return {
            'user_query': user_query,
            'intent': intent.value,
            'confidence': confidence,
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
        context = self.context_manager.get_current_context()
        status = {
            'session_id': context['session_id'],
            'current_dataset': context['current_dataset'],
            'conversations': context['conversation_count']
        }
        
        if self.context_manager.current_dataset:
            file_info = self.file_processor.process_file(self.context_manager.current_dataset)
            if file_info['type'] == 'dataframe':
                status['file_type'] = 'dataframe'
                status['shape'] = file_info['shape']
                status['columns'] = file_info['columns']
            elif file_info['type'] == 'text':
                status['file_type'] = 'text'
                status['word_count'] = file_info['word_count']
            elif file_info['type'] == 'image':
                status['file_type'] = 'image'
                status['image_size'] = file_info['image_size']
        
        return status


