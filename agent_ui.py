import streamlit as st
import os
import logging
from dotenv import load_dotenv
import uuid

# Import agent components
from langgraph_agent import invoke_agent

# Setup
load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Streamlit config
st.set_page_config(page_title="Text2SQL Agent", layout="wide")

# Initialize session state
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "actor_id" not in st.session_state:
    st.session_state.actor_id = "user-tanpat"

# Header
st.title("🚀 Text2SQL Agent with Memory")
st.markdown("Chat with AI to query your pet food sales database")

# Sidebar - Info
with st.sidebar:
    st.subheader("📋 Session Info")
    st.write(f"**Session ID**: `{st.session_state.session_id[:8]}...`")
    st.write(f"**Actor**: `{st.session_state.actor_id}`")
    st.write(f"**Messages**: {len(st.session_state.chat_history)}")
    
    if st.button("🔄 Reset Session", use_container_width=True):
        st.session_state.session_id = str(uuid.uuid4())
        st.session_state.chat_history = []
        st.rerun()

# Main chat area
st.subheader("💬 Conversation")

# Display chat history
chat_container = st.container()
with chat_container:
    for msg in st.session_state.chat_history:
        if msg["role"] == "user":
            with st.chat_message("user"):
                st.write(msg["content"])
        else:
            with st.chat_message("assistant"):
                st.write(msg["content"])

# Input area
st.divider()
user_input = st.chat_input("Ask me about your database...", key="chat_input")

if user_input:
    # Add user message to history
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    
    # Display user message
    with st.chat_message("user"):
        st.write(user_input)
    
    # Show thinking status
    with st.chat_message("assistant"):
        status_placeholder = st.empty()
        status_placeholder.info("🤔 Agent is thinking...")
        
        try:
            # Call agent
            response = invoke_agent(
                question=user_input,
                actor_id=st.session_state.actor_id,
                session_id=st.session_state.session_id,
                reset_memory=False
            )
            
            # Update status
            status_placeholder.empty()
            
            # Display response
            st.write(response)
            
            # Add to history
            st.session_state.chat_history.append({"role": "assistant", "content": response})
            
        except Exception as e:
            status_placeholder.error(f"❌ Error: {str(e)}")
            st.session_state.chat_history.append({"role": "assistant", "content": f"Error: {str(e)}"})
