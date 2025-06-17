import streamlit as st
import os
from dotenv import load_dotenv
from agent_core import IntelligentAgent
from file_processor import FileProcessor
from data_preprocessor import DataPreprocessor
import pandas as pd

# Set page config
st.set_page_config(
    page_title="InsightX - AI Data Analysis Agent",
    page_icon="📊",
    layout="centered"
)

# Initialize session state
#Writing the if condtions for the session state is important because 
# every time the app reruns in a single user's session,the session state is reset
# so we need to check if the session state is already set and if not, we need to set it
# this is important because if we don't do this, the app will crash
# this is because the session state is not initialized and the app will try to access it
# and it will crash
#these if conditions will run only once because they are inserting the keys and setting their
#to none. 

if 'agent' not in st.session_state:
    st.session_state.agent = None
if 'current_dataset' not in st.session_state:
    st.session_state.current_dataset = None
if 'processed_data' not in st.session_state:
    st.session_state.processed_data = None
if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = []

# Load environment variables
env_path = "Configs/.env"
if os.path.exists(env_path):
    load_dotenv(env_path)
    TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")
else:
    TOGETHER_API_KEY = None

def initialize_agent():
    """Initialize the agent with API key"""
    if TOGETHER_API_KEY:
        st.session_state.agent = IntelligentAgent(TOGETHER_API_KEY)
        return True
    return False

def process_dataset(file_path):
    """Process the selected dataset"""
    success = False #Flags whether the entire process completed without errors. It's initialized to False and set to True only upon successful completion.
    message = "Unknown error during dataset processing."
    processed_data_output = None # Initialize processed_data_output

    try:
        file_processor = FileProcessor()
        file_info = file_processor.process_file(file_path)
        
        if not isinstance(file_info, dict):
            message = "File processing returned an unexpected non-dictionary object."
            return success, message, processed_data_output # Return all three values
        
        st.session_state.agent.register_dataset(file_path, file_info['type'])
            
        if file_info['type'] == 'dataframe':
            df = file_info['data']
            # Preprocess the DataFrame for Streamlit compatibility
            data_preprocessor = DataPreprocessor(agent=st.session_state.agent)  #passing the instance of the agent because there is a function get_column_types which requires llm recommendations
            processed_df, preproc_message = data_preprocessor.preprocess_dataframe(df)
            st.session_state.processed_data = processed_df

            processed_data_output = processed_df # Store for returning
            success = True
            message = f"Dataset processed successfully. Shape: {processed_df.shape}. {preproc_message}"
        else:
            st.session_state.processed_data = file_info['data'] # Store raw data for non-dataframes
            success = True
            message = f"File processed successfully. Type: {file_info['type']}"
            processed_data_output = file_info['data'] # Store raw data for returning

    except Exception as e:
        message = f"Error processing dataset: {str(e)}"
    
    return success, message, processed_data_output # Return all three values

def display_dataframe_info(df):
    """Display dataframe information"""
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("Dataset Shape:", df.shape)
        st.write("Columns:", ", ".join(df.columns))
    
    with col2:
        st.write("Data Types:")
        st.write(df.dtypes)

def display_conversation_history():
    """Display conversation history"""
    for entry in st.session_state.conversation_history:
        with st.chat_message("user"):
            st.write(entry['user_query'])
        with st.chat_message("assistant"):
            st.write(entry['agent_response'])
            if entry.get('visualization'):
                st.image(entry['visualization'])

def main():
    st.title("📊InsightX - AI Data Analyst Agent")
    
    # Sidebar for dataset selection and controls
    with st.sidebar:
        st.header("Dataset Selection",divider="rainbow")
        
        # Ensure agent is initialized if not already done and API key is present
        if st.session_state.agent is None and TOGETHER_API_KEY:
            if initialize_agent():
                st.success("Agent initialized successfully!")
            else:
                st.error("Failed to initialize agent. Check API key.")
        
        # Optional: Button to re-initialize agent (e.g., if API key changes or for debugging)
        if st.session_state.agent is not None:
            if st.button("Reinitialize Agent"):
                st.session_state.agent = None # Reset to trigger re-initialization on next rerun
                st.session_state.conversation_history = [] # Clear conversation history
                st.session_state.current_dataset = None # Clear current dataset
                st.session_state.processed_data = None # Clear processed data
                if initialize_agent():
                    st.success("Agent reinitialized successfully!")
                else:
                    st.error("Failed to reinitialize agent. Check API key.")

        # Dataset selection
        dataset_files = [f for f in os.listdir("Datasets") if os.path.isfile(os.path.join("Datasets", f))]
        selected_file = st.selectbox("Select Dataset", dataset_files)
        
        if selected_file and st.button("Load Dataset"):
            if st.session_state.agent is None:
                st.error("Agent not initialized. Please ensure your API key is set and try refreshing.")
                return
            file_path = os.path.join("Datasets", selected_file)
            
            # Unpack three values here now
            success, message, data_output = process_dataset(file_path) 
            
            if success:
                st.success(message)
                st.session_state.current_dataset = file_path
                # Update processed_data in session state if it's a dataframe
                if data_output is not None:
                    st.session_state.processed_data = data_output
            else:
                st.error(message)
        
        # Session controls
        st.header("Session Controls")
        if st.button("Save Session"):
            if st.session_state.agent:
                st.session_state.agent.save_session()
                st.success("Session saved successfully!")
        
        if st.button("Load Session"):
            if st.session_state.agent:
                st.session_state.agent.context_manager.load_session()
                st.success("Session loaded successfully!")
    
    # Main content area
    if st.session_state.current_dataset and st.session_state.agent:
        st.header("Dataset Information")
        if isinstance(st.session_state.processed_data, pd.DataFrame):
            display_dataframe_info(st.session_state.processed_data)
        elif st.session_state.processed_data is not None:
            st.write(f"File Type: {st.session_state.agent.context_manager.dataset_registry[os.path.basename(st.session_state.current_dataset)].data_types['content'] if os.path.basename(st.session_state.current_dataset) in st.session_state.agent.context_manager.dataset_registry else 'text/image'}")
            st.write("Content Preview:")
            st.text(str(st.session_state.processed_data)[:500] + "...") # Display a preview for text/image data
        else:
            st.info("No data loaded for display. Please load a dataset.")

        # Query input
        st.header("Ask Questions")
        user_query = st.chat_input("Ask a question about your data...")
        
        if user_query:
            # Process query
            response = st.session_state.agent.process_query(user_query)
            
            # Add to conversation history
            st.session_state.conversation_history.append({
                'user_query': user_query,
                'agent_response': response['llm_response'],
                'visualization': response.get('visualization')
            })
            
            # Display conversation
            display_conversation_history()
    else:
        st.info("Please select and load a dataset from the sidebar to begin.")

if __name__ == "__main__":
    main() 