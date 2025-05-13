import streamlit as st
import time

def generate_response_stream(prompt):
    """Simulates a streaming response from a prompt-based model."""
    for word in prompt.split():
        yield word + " "
        time.sleep(0.1)  # Simulate processing time

st.title("Streaming Prompt Response")

prompt_text = st.text_area("Enter your prompt:", "Type here...")

if st.button("Submit"):
    if prompt_text:
        message_placeholder = st.empty()
        full_response = ""
        for response_chunk in generate_response_stream(prompt_text):
            full_response += response_chunk
            message_placeholder.markdown(full_response + "▌")
        message_placeholder.markdown(full_response)
    else:
        st.warning("Please enter a prompt.")