"""
GreenGov AI: Discovering Clean Energy, Water Conservation & Sustainable Agriculture Subsidies
Main Streamlit Application
"""

import streamlit as st
from dotenv import load_dotenv
import os
from pathlib import Path
import google.generativeai as genai

from rag import RAGPipeline
from agents import run_agentic_workflow, ProfileAgent

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="GreenGov AI - Sustainable Subsidy Discovery",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
    }
    .user-message {
        background-color: #e3f2fd;
        border-left: 4px solid #2196f3;
    }
    .assistant-message {
        background-color: #f5f5f5;
        border-left: 4px solid #4caf50;
    }
    .sidebar-header {
        font-size: 1.2rem;
        font-weight: bold;
        margin-top: 1rem;
        margin-bottom: 0.5rem;
    }
    .scheme-item {
        padding: 0.5rem;
        margin: 0.25rem 0;
        background-color: #f0f7ff;
        border-radius: 5px;
        font-size: 0.9rem;
    }
    .disclaimer {
        font-size: 0.8rem;
        color: #666;
        font-style: italic;
        border-top: 1px solid #ddd;
        margin-top: 2rem;
        padding-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'rag_pipeline' not in st.session_state:
    st.session_state.rag_pipeline = None
if 'initialized' not in st.session_state:
    st.session_state.initialized = False

def initialize_rag_pipeline():
    """Initialize RAG pipeline with API key and load documents"""
    api_key = os.getenv("GOOGLE_API_KEY")
    
    if not api_key:
        st.error("⚠️ Google API Key not found. Please set GOOGLE_API_KEY in .env file")
        return False
    
    try:
        # Initialize RAG pipeline
        rag = RAGPipeline(api_key=api_key, data_path="./data")
        
        # Load or create vector store
        with st.spinner("📚 Loading sustainability schemes and creating knowledge base..."):
            rag.load_or_create_index(index_path="./faiss_index")
        
        st.session_state.rag_pipeline = rag
        st.session_state.initialized = True
        return True
        
    except Exception as e:
        st.error(f"❌ Failed to initialize RAG pipeline: {str(e)}")
        return False

def display_sidebar():
    """Display sidebar with scheme information"""
    with st.sidebar:
        st.image("https://img.icons8.com/color/96/000000/greenhouse.png", width=80)
        st.title("🌿 GreenGov AI")
        st.markdown("*Discover Sustainability Subsidies*")
        
        st.markdown("---")
        
        st.markdown("## 📋 Supported Schemes")
        
        if st.session_state.rag_pipeline and st.session_state.rag_pipeline.documents:
            schemes = st.session_state.rag_pipeline.get_scheme_names()
            
            if schemes:
                for scheme in schemes:
                    st.markdown(f"""
                    <div class="scheme-item">
                        ✅ {scheme}
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("Loading schemes from documents...")
        else:
            st.info("No documents loaded yet. Check data directory.")
        
        st.markdown("---")
        st.markdown("## 🎯 How It Works")
        st.markdown("""
        1. **Profile Analysis** - AI extracts your details
        2. **Document Retrieval** - Finds relevant schemes
        3. **Eligibility Check** - Analyzes your fit
        4. **Response Generation** - Provides actionable info
        """)
        
        st.markdown("---")
        st.markdown("## 💡 Example Questions")
        examples = [
            "I'm a small farmer in Punjab with 2 acres, want solar pumps",
            "How can I get subsidies for drip irrigation in Maharashtra?",
            "What crop insurance schemes are available for rice farmers?",
            "I need soil health card for my organic farm in Kerala"
        ]
        
        for example in examples:
            if st.button(example, key=example[:20]):
                st.session_state.user_input = example
                st.rerun()
        
        st.markdown("---")
        st.markdown('<div class="disclaimer">⚠️ Always verify through official government portals before applying.</div>', 
                   unsafe_allow_html=True)

def main():
    """Main application entry point"""
    
    # Display header
    st.markdown("""
    <div class="main-header">
        <h1 style="color: white; margin: 0;">🌿 GreenGov AI</h1>
        <p style="color: white; margin: 0; opacity: 0.9;">Discovering Clean Energy, Water Conservation & Sustainable Agriculture Subsidies</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    display_sidebar()
    
    # Initialize RAG pipeline on first run
    if not st.session_state.initialized:
        with st.spinner("🚀 Initializing GreenGov AI..."):
            if initialize_rag_pipeline():
                st.success("✅ GreenGov AI is ready! Ask about sustainability schemes.")
            else:
                st.stop()
    
    # Main chat interface
    chat_container = st.container()
    
    with chat_container:
        # Display chat history
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        # Chat input
        if prompt := st.chat_input("Ask about sustainability schemes, subsidies, and eligibility..."):
            
            # Add user message to history
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Generate response
            with st.chat_message("assistant"):
                with st.spinner("🤖 Analyzing your query with GreenGov AI..."):
                    try:
                        # Initialize Gemini model
                        api_key = os.getenv("GOOGLE_API_KEY")
                        genai.configure(api_key=api_key)
                        model = genai.GenerativeModel('gemini-pro')
                        
                        # Run agentic workflow
                        result = run_agentic_workflow(
                            query=prompt,
                            rag_pipeline=st.session_state.rag_pipeline,
                            model=model,
                            k=5
                        )
                        
                        # Display response
                        st.markdown(result["response"])
                        
                        # Optionally display profile info in expander
                        if result["profile"] and any(result["profile"].values()):
                            with st.expander("📊 Profile Extracted"):
                                profile_agent = ProfileAgent(model)
                                st.markdown(profile_agent.format_profile_prompt(result["profile"]))
                        
                        # Add to history
                        st.session_state.messages.append({"role": "assistant", "content": result["response"]})
                        
                    except Exception as e:
                        error_msg = f"❌ Error: {str(e)}\n\nPlease check your API key and document files."
                        st.error(error_msg)
                        st.session_state.messages.append({"role": "assistant", "content": error_msg})
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; font-size: 0.8rem;">
        <p>🌱 GreenGov AI uses RAG (Retrieval Augmented Generation) and Agentic AI to provide scheme recommendations.</p>
        <p>Powered by Google Gemini API, FAISS, and LangChain</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
