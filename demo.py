import streamlit as st
import time
from typing import Generator
import re

MAX_PROMPT_LENGTH = 1000
MAX_PROCESSING_TIME = 30  # seconds
WORD_DELAY = 0.1  # seconds

def sanitize_input(text: str) -> str:
    """Sanitize user input by removing potentially harmful characters."""
    # Remove any HTML/markdown special characters
    return re.sub(r'[<>&;{}]', '', text)

def validate_prompt(prompt: str) -> tuple[bool, str]:
    """Validate the prompt text."""
    if not prompt or prompt.isspace():
        return False, "Please enter a non-empty prompt."
    if len(prompt) > MAX_PROMPT_LENGTH:
        return False, f"Prompt exceeds maximum length of {MAX_PROMPT_LENGTH} characters."
    return True, ""

def generate_response_stream(prompt: str) -> Generator[str, None, None]:
    """Simulates a streaming response from a prompt-based model with safety limits."""
    start_time = time.time()
    try:
        for word in prompt.split():
            if time.time() - start_time > MAX_PROCESSING_TIME:
                yield "\n\n[Processing timeout reached]"
                break
            yield word + " "
            time.sleep(WORD_DELAY)
    except Exception as e:
        yield f"\n\n[Error during processing: {str(e)}]"

st.title("Streaming Prompt Response")

prompt_text = st.text_area("Enter your prompt:", "Type here...")

if st.button("Submit"):
    try:
        # Validate and sanitize input
        is_valid, error_message = validate_prompt(prompt_text)
        if not is_valid:
            st.warning(error_message)
        else:
            sanitized_prompt = sanitize_input(prompt_text)
            message_placeholder = st.empty()
            full_response = ""
            
            with st.spinner("Processing..."):
                for response_chunk in generate_response_stream(sanitized_prompt):
                    full_response += response_chunk
                    message_placeholder.markdown(full_response + "▌")
                message_placeholder.markdown(full_response)
                
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")