```python
import streamlit as st
import time
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def generate_response_stream(prompt):
    """Simulates a streaming response with input validation"""
    try:
        if not prompt or len(prompt) > 500:
            raise ValueError("Invalid prompt length")
        
        for word in prompt.split():
            yield word + " "
            time.sleep(0.1)  # Simulate processing time
    except Exception as e:
        logger.error(f"Response generation error: {e}")
        yield "Error generating response"

def main():
    st.title("Streaming Prompt Response")

    prompt_text = st.text_area("Enter your prompt:", "Type here...")

    if st.button("Submit"):
        if prompt_text and len(prompt_text) <= 500:
            message_placeholder = st.empty()
            full_response = ""
            
            try:
                for response_chunk in generate_response_stream(prompt_text):
                    full_response += response_chunk
                    message_placeholder.markdown(full_response + "\u258c")
                
                message_placeholder.markdown(full_response)
            except Exception as e:
                st.error(f"An error occurred: {e}")
        else:
            st.warning("Please enter a valid prompt (1-500 characters).")

if __name__ == "__main__":
    main()
```