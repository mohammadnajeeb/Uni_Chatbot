"""
Streamlit web app for the Faiz Chatbot.
"""
import os
import sys
import logging
import streamlit as st
from dotenv import load_dotenv

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Load environment variables
load_dotenv()

# Import the chatbot
from chatbot.faiz import FaizChatbot

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize session state
def init_session_state():
    """Initialize the session state for the chat interface."""
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    if "chatbot" not in st.session_state:
        try:
            st.session_state.chatbot = FaizChatbot(
                model_name=st.session_state.get("model_name", os.getenv('CHAT_MODEL')),
                embedding_model=st.session_state.get("embedding_model", os.getenv('EMBEDDING_MODEL')),
                temperature=st.session_state.get("temperature", 0.7)
            )
            logger.info("Chatbot initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing chatbot: {e}")
            st.error(f"Failed to initialize chatbot: {e}")

def run_app():
    """Run the Streamlit app."""
    # Set page config
    st.set_page_config(
        page_title="AMUBot - Aligarh Muslim University Assistant",
        page_icon="🎓",
        layout="wide"
    )
    
    # Initialize session state
    init_session_state()
    
    # Header
    st.title("🎓 AMUBot - Your Aligarh Muslim University Guide")
    
    # Sidebar with settings
    with st.sidebar:
        st.header("About")
        st.write("""
        AMUBot is your personal assistant for all things related to Aligarh Muslim University (AMU).
        Ask me anything about AMU's programs, campus, admissions, exams, or other general information.
        """)
        
        st.header("Settings")
        
        model_options = {
            "mistralai/Mistral-7B-Instruct-v0.2": "Mistral 7B Instruct",
            "google/flan-t5-base": "Flan-T5 Base",
            "tiiuae/falcon-7b-instruct": "Falcon 7B Instruct"
        }
        
        model_name = st.selectbox(
            "Chat Model", 
            list(model_options.keys()), 
            format_func=lambda x: model_options.get(x, x)
        )
        
        embedding_options = {
            "sentence-transformers/all-MiniLM-L6-v2": "MiniLM-L6",
            "sentence-transformers/all-mpnet-base-v2": "MPNet Base",
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2": "Multilingual MiniLM"
        }
        
        embedding_model = st.selectbox(
            "Embedding Model", 
            list(embedding_options.keys()), 
            format_func=lambda x: embedding_options.get(x, x)
        )
        
        temperature = st.slider("Creativity", min_value=0.0, max_value=1.0, value=0.7, step=0.1)
        
        # Update settings
        if (model_name != st.session_state.get("model_name") or 
            embedding_model != st.session_state.get("embedding_model") or
            temperature != st.session_state.get("temperature")):
            st.session_state.model_name = model_name
            st.session_state.embedding_model = embedding_model
            st.session_state.temperature = temperature
            # Reinitialize chatbot with new settings
            st.session_state.pop("chatbot", None)
            init_session_state()
        
        st.divider()
        
        st.write("Developed by Mohammad Najeeb")
        st.write("© 2023 Faiz Chatbot")
        
        # Add a note about Hugging Face
        st.info("This app uses free Hugging Face models. First-time queries may be slower as models are loaded.")
    
    # Chat interface
    chat_container = st.container()
    
    # Input form
    with st.form(key="chat_form", clear_on_submit=True):
        user_input = st.text_area("Your question:", height=100)
        submit_button = st.form_submit_button("Ask AMUBot")
    
    # Process input
    if submit_button and user_input:
        # Add user message to chat history
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        
        # Get response from chatbot
        try:
            with st.spinner("AMUBot is thinking... (This may take a while for the first query as models load)"):
                response = st.session_state.chatbot.answer(user_input)
            
            # Add chatbot response to chat history
            st.session_state.chat_history.append({"role": "assistant", "content": response})
        except Exception as e:
            logger.error(f"Error getting response: {e}")
            st.error(f"Error: {str(e)}")
            # Add error details
            with st.expander("Error details"):
                import traceback
                st.code(traceback.format_exc())
    
    # Display chat history
    with chat_container:
        for message in st.session_state.chat_history:
            if message["role"] == "user":
                st.markdown(f'<div style="background-color:#2b5797;color:white;padding:10px;border-radius:5px;margin-bottom:10px;"><p style="font-weight:bold;">You</p>{message["content"]}</div>', unsafe_allow_html=True)
            else:  # assistant
                st.markdown(f'<div style="background-color:#1e7145;color:white;padding:10px;border-radius:5px;margin-bottom:10px;"><p style="font-weight:bold;">AMUBot</p>{message["content"]}</div>', unsafe_allow_html=True)
    
    # Footer
    st.markdown("""
    ---
    **Note**: AMUBot provides information based on data from AMU websites. For official information, please visit the [AMU website](https://www.amu.ac.in/) or [AMU Controller of Exams](https://amucontrollerexams.com/).
    
    Powered by [Hugging Face](https://huggingface.co/) open-source AI models.
    """)

if __name__ == "__main__":
    run_app()