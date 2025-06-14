import os
import time
import base64
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import io
from datetime import datetime
from dotenv import load_dotenv
import traceback
import sys

from agent_core import IntelligentAgent
from file_processor import FileProcessor
from data_preprocessor import DataPreprocessor

# Create required directories
for dir_name in ["Graph_Plots", "Saved_Sessions", "Datasets", "Configs"]:
    os.makedirs(dir_name, exist_ok=True)

# Load environment variables
env_path = os.path.join("Configs", ".env")
if os.path.exists(env_path):
    load_dotenv(env_path)
else:
    print("Warning: .env file not found in Configs folder")

def save_visualization(base64_image: str, query: str):
    """Save base64 encoded image to file"""
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_query = "".join(c if c.isalnum() else "_" for c in query[:30])
        filename = os.path.join("Graph_Plots", f"viz_{timestamp}_{safe_query}.png")
        
        if ',' in base64_image:
            base64_image = base64_image.split(',')[1]
        
        image_data = base64.b64decode(base64_image)
        image = Image.open(io.BytesIO(image_data))
        image = image.convert('RGB')
        image.save(filename, 'PNG')
        print(f"\nVisualization saved as: {os.path.abspath(filename)}")
    except Exception as e:
        print(f"Error saving visualization: {str(e)}")

def process_dataset(file_processor: FileProcessor, data_preprocessor: DataPreprocessor, dataset_path: str):
    """Process dataset and return file info"""
    print(f"\nProcessing file: {dataset_path}")
    file_info = file_processor.process_file(dataset_path)
    print(f"File processed successfully. Type: {file_info['type']}")
    
    if file_info['type'] == 'dataframe':
        if 'message' in file_info:
            print(f"\n{file_info['message']}")
        print(f"Data shape: {file_info['shape']}")
        print(f"Columns: {', '.join(file_info['columns'])}")
        
        # Preprocess dataframe
        df = file_info['data']
        cleaned_df = data_preprocessor.preprocess_dataframe(df)
        analysis = data_preprocessor.analyze_structure(cleaned_df)
        
        print("\nPreprocessing Summary:")
        print(f"Original shape: {df.shape}")
        print(f"Cleaned shape: {cleaned_df.shape}")
        print(f"Numeric columns: {len(analysis['numeric_columns'])}")
        print(f"Categorical columns: {len(analysis['categorical_columns'])}")
        print(f"Datetime columns: {len(analysis['datetime_columns'])}")
    else:
        print("Processing completed.")
    
    return file_info

def main():
    try:
        # Check for API key
        api_key = os.getenv('TOGETHER_API_KEY')
        if not api_key:
            print("Error: TOGETHER_API_KEY not found in environment variables.")
            return

        # Initialize processors
        file_processor = FileProcessor()
        data_preprocessor = DataPreprocessor()
        
        # Get supported files
        supported_files = [f for f in os.listdir("Datasets") 
                         if any(f.endswith(ext) for ext in file_processor.supported_formats)]
        
        if not supported_files:
            print("Error: No supported files found in Datasets folder.")
            print("Supported formats:", ", ".join(file_processor.supported_formats))
            return
        
        # Display available files
        print("\nAvailable files in Datasets folder:")
        for i, file in enumerate(supported_files, 1):
            print(f"{i}. {file}")
        
        # Get user choice
        while True:
            choice = input("\nEnter the number of the file you want to process (or 'exit' to quit): ").strip()
            if choice.lower() == 'exit':
                return
            
            try:
                choice_num = int(choice)
                if 1 <= choice_num <= len(supported_files):
                    selected_file = supported_files[choice_num - 1]
                    break
                print(f"Please enter a number between 1 and {len(supported_files)}")
            except ValueError:
                print("Please enter a valid number")
        
        # Process selected file
        dataset_path = os.path.join("Datasets", selected_file)
        file_info = process_dataset(file_processor, data_preprocessor, dataset_path)
        
        # Initialize agent
        print("\nInitializing agent...")
        agent = IntelligentAgent(together_api_key=api_key)
        agent.register_dataset(dataset_path, os.path.splitext(dataset_path)[1])
        print("New session created and dataset registered successfully")
        
        # Main interaction loop
        print("\n=== Intelligent Agent Ready ===")
        print("Available commands:")
        print("1. 'query <your question>' - Ask a question about the data")
        print("2. 'save' - Save current session")
        print("3. 'load <session_number>' - Load a specific session")
        print("4. 'list' - List all saved sessions")
        print("5. 'status' - Show current agent status")
        print("6. 'history' - Show conversation history")
        print("7. 'exit' or 'quit' - End the session")
        print("8. 'new' - Start a new session with a different dataset")
        
        while True:
            user_input = input("\nCommand: ").strip()
            
            if user_input.lower() in ['exit', 'quit']:
                print("\nEnding session. Goodbye!")
                break
            
            if not user_input:
                continue
            
            if user_input.lower() == 'new':
                print("\nStarting new session...")
                main()
                return
            
            elif user_input.lower().startswith('query '):
                query = user_input[6:].strip()
                if query:
                    print("\nProcessing your query...")
                    result = agent.process_query(query)
                    print("\nAgent's Response:")
                    print(result['llm_response'])
                    if result['visualization']:
                        save_visualization(result['visualization'], query)
            
            elif user_input.lower() == 'save':
                agent.save_session()
            
            elif user_input.lower().startswith('load '):
                try:
                    session_num = int(user_input[5:].strip())
                    agent.context_manager.load_session(os.path.join("Saved_Sessions", f"session_state{session_num}.pkl"))
                except ValueError:
                    print("Please provide a valid session number.")
            
            elif user_input.lower() == 'list':
                session_files = [f for f in os.listdir("Saved_Sessions") 
                               if f.startswith("session_state") and f.endswith(".pkl")]
                if session_files:
                    print("\nAvailable sessions:")
                    for file in sorted(session_files, key=lambda x: int(''.join(filter(str.isdigit, x)))):
                        print(f"- {file}")
                else:
                    print("No saved sessions found.")
            
            elif user_input.lower() == 'status':
                status = agent.get_agent_status()
                print("\nAgent Status:")
                print(f"Session ID: {status['session_id']}")
                print(f"Current Dataset: {status['current_dataset']}")
                print(f"Conversations: {status['conversations']}")
            
            elif user_input.lower() == 'history':
                history = agent.get_conversation_history()
                if history:
                    print("\nConversation History (Last 3 conversations):")
                    for entry in history[-3:]:
                        print(f"\nTime: {entry['timestamp']}")
                        print(f"Query: {entry['user_query']}")
                        print(f"Intent: {entry['intent']}")
                        print(f"Response: {entry['agent_response']}")
                        print("-" * 50)
                else:
                    print("No conversation history available.")
            
            else:
                print("Unknown command. Please use one of the available commands.")
            
            print("\n" + "="*50)
            
    except Exception as e:
        print(f"Critical error: {str(e)}")
        traceback.print_exc()

if __name__ == "__main__":
    main() 