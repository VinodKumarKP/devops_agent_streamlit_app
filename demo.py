import streamlit as st
import time
from typing import Generator

def validate_prompt(prompt: str) -> bool:
    """Validate the prompt input."""
    if not prompt or len(prompt.strip()) == 0:
        return False
    if len(prompt) > 500:  # Set reasonable limit
        return False
    return True

def generate_response_stream(prompt: str) -> Generator[str, None, None]:
    """Simulates a streaming response from a prompt-based model."""
    for word in prompt.split():
        yield word + " "
        time.sleep(0.1)  # Simulate processing time

def main():
    st.title("Streaming Prompt Response")
    
    prompt_text = st.text_area(
        "Enter your prompt:",
        "Type here...",
        max_chars=500,
        help="Enter your prompt (max 500 characters)"
    )

    if st.button("Submit"):
        if validate_prompt(prompt_text):
            message_placeholder = st.empty()
            full_response = ""
            try:
                for response_chunk in generate_response_stream(prompt_text):
                    full_response += response_chunk
                    message_placeholder.markdown(full_response + "▌")
                message_placeholder.markdown(full_response)
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
        else:
            st.warning("Please enter a valid prompt (1-500 characters).")

if __name__ == "__main__":
    main()