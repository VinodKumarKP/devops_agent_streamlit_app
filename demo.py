import streamlit as st
import time
from typing import Generator
import re

MAX_PROMPT_LENGTH = 1000  # Maximum allowed prompt length
MAX_PROCESSING_TIME = 30  # Maximum processing time in seconds

def sanitize_input(text: str) -> str:
    """Sanitize user input to prevent XSS and other injection attacks."""
    # Remove any potentially harmful markdown or HTML
    text = re.sub(r'[<>]', '', text)
    return text.strip()

def validate_prompt(prompt: str) -> tuple[bool, str]:
    """Validate the prompt text."""
    if not prompt or prompt.isspace():
        return False, "Please enter a prompt."
    if len(prompt) > MAX_PROMPT_LENGTH:
        return False, f"Prompt exceeds maximum length of {MAX_PROMPT_LENGTH} characters."
    return True, ""

def generate_response_stream(prompt: str) -> Generator[str, None, None]:
    """Simulates a streaming response from a prompt-based model with timeout."""
    start_time = time.time()
    try:
        for word in prompt.split():
            if time.time() - start_time > MAX_PROCESSING_TIME:
                yield "\n\nProcessing timeout reached."
                break
            yield word + " "
            time.sleep(0.1)  # Simulate processing time
    except Exception as e:
        yield f"\n\nError during processing: {str(e)}"

st.title("Streaming Prompt Response")

prompt_text = st.text_area("Enter your prompt:", "Type here...")

if st.button("Submit"):
    try:
        # Validate and sanitize input
        sanitized_prompt = sanitize_input(prompt_text)
        is_valid, error_msg = validate_prompt(sanitized_prompt)
        
        if not is_valid:
            st.warning(error_msg)
        else:
            message_placeholder = st.empty()
            full_response = ""
            
            for response_chunk in generate_response_stream(sanitized_prompt):
                full_response += response_chunk
                message_placeholder.markdown(full_response + "▌")
            
            message_placeholder.markdown(full_response)
    
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")