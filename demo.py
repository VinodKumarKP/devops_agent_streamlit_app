import streamlit as st
import time
from typing import Generator
import re

MAX_PROMPT_LENGTH = 1000
MAX_PROCESSING_TIME = 30  # seconds

def sanitize_input(text: str) -> str:
    """Sanitize user input by removing potentially dangerous content."""
    # Remove any HTML/markdown
    text = re.sub(r'<[^>]*>', '', text)
    return text.strip()

def generate_response_stream(prompt: str) -> Generator[str, None, None]:
    """Simulates a streaming response from a prompt-based model with safety limits."""
    start_time = time.time()
    try:
        for word in prompt.split():
            if time.time() - start_time > MAX_PROCESSING_TIME:
                yield "\n\nProcessing time limit exceeded."
                break
            yield word + " "
            time.sleep(0.1)  # Simulate processing time
    except Exception as e:
        yield f"\n\nError during processing: {str(e)}"

st.title("Streaming Prompt Response")

prompt_text = st.text_area("Enter your prompt:", "Type here...", 
                          max_chars=MAX_PROMPT_LENGTH)

if st.button("Submit"):
    try:
        if not prompt_text or prompt_text == "Type here...":
            st.warning("Please enter a prompt.")
        else:
            # Sanitize input
            sanitized_prompt = sanitize_input(prompt_text)
            if len(sanitized_prompt) < 3:
                st.warning("Prompt is too short. Please enter a longer prompt.")
                st.stop()
                
            message_placeholder = st.empty()
            full_response = ""
            
            with st.spinner("Processing..."):
                for response_chunk in generate_response_stream(sanitized_prompt):
                    full_response += response_chunk
                    message_placeholder.markdown(full_response + "▌")
                    
            message_placeholder.markdown(full_response)
            
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")