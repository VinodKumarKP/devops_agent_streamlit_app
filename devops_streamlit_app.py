import json
import queue
import threading
import time
import uuid
import os
from datetime import datetime
from typing import Optional

import boto3
import streamlit as st
from streamlit.runtime.scriptrunner import get_script_run_ctx, add_script_run_ctx
from time import sleep

# Constants
MAX_QUEUE_SIZE = 1000
MAX_PROMPT_LENGTH = 4000
REGION_NAME = os.getenv('AWS_REGION', 'us-east-1')
AGENT_ALIAS_ID = os.getenv('BEDROCK_AGENT_ALIAS_ID')
AGENT_ID = os.getenv('BEDROCK_AGENT_ID')

class BedrockChatApp:
    def __init__(self):
        self.initialize_state()
        self.configure_page()
        self._validate_aws_config()

    def _validate_aws_config(self):
        """Validate AWS configuration and credentials"""
        if not AGENT_ALIAS_ID or not AGENT_ID:
            raise ValueError("Missing required AWS Bedrock configuration")
        
        try:
            self.bedrock_client = boto3.client('bedrock-agent-runtime', region_name=REGION_NAME)
        except Exception as e:
            raise RuntimeError(f"Failed to initialize AWS Bedrock client: {str(e)}")

    def initialize_state(self):
        """Initialize all session state variables"""
        state_vars = {
            "user_id": str(uuid.uuid4()),
            "conversation_history": {},
            "is_authenticated": False,
            "is_processing": False,
            "user_database": {},
            "response_queue": queue.Queue(maxsize=MAX_QUEUE_SIZE),
            "waiting_for_response": False,
            "session_id": str(uuid.uuid4())
        }
        
        for var, value in state_vars.items():
            if var not in st.session_state:
                st.session_state[var] = value

    def configure_page(self):
        """Configure Streamlit page settings"""
        st.set_page_config(
            page_title="AWS Bedrock Chat",
            page_icon="🤖",
            layout="wide"
        )

    def validate_prompt(self, prompt: str) -> bool:
        """Validate user input prompt"""
        if not prompt or len(prompt) > MAX_PROMPT_LENGTH:
            return False
        # Add additional validation rules as needed
        return True

    def invoke_bedrock_model_with_streaming(self, prompt: str, user_id: str, session_id: str) -> Optional[str]:
        """Invoke AWS Bedrock model with streaming response"""
        if not self.validate_prompt(prompt):
            raise ValueError("Invalid prompt")

        try:
            response = self.bedrock_client.invoke_agent(
                agentAliasId=AGENT_ALIAS_ID,
                agentId=AGENT_ID,
                enableTrace=False,
                endSession=False,
                inputText=prompt,
                sessionId=session_id,
                streamingConfigurations={'streamFinalResponse': True}
            )

            full_response = ''
            if response.get('completion'):
                for event in response['completion']:
                    text_chunk = ''
                    if "chunk" in event:
                        chunk = event["chunk"]
                        text_chunk = chunk.get("bytes").decode()

                    if text_chunk:
                        full_response += text_chunk
                        with threading.Lock():
                            st.session_state.response_queue.put((user_id, text_chunk, False))

            return full_response

        except Exception as e:
            error_msg = f"Error: {str(e)}"
            st.session_state.response_queue.put((user_id, error_msg, True))
            return None

    def process_request(self, prompt: str, user_id: str, session_id: str):
        """Process the user request and get a response from AWS Bedrock"""
        try:
            with threading.Lock():
                if user_id not in st.session_state.conversation_history:
                    st.session_state.conversation_history[user_id] = []

                st.session_state.conversation_history[user_id].append({
                    "role": "user", 
                    "content": prompt,
                    "timestamp": datetime.now().isoformat()
                })
                st.session_state.waiting_for_response = True

            full_response = self.invoke_bedrock_model_with_streaming(prompt, user_id, session_id)
            
            if full_response:
                with threading.Lock():
                    st.session_state.conversation_history[user_id].append({
                        "role": "assistant",
                        "content": full_response,
                        "timestamp": datetime.now().isoformat()
                    })

        except Exception as e:
            st.error(f"Error processing request: {str(e)}")
        finally:
            with threading.Lock():
                st.session_state.is_processing = False
                st.session_state.waiting_for_response = False

    def chat_interface(self):
        """Display chat interface"""
        st.title("AWS Bedrock Chat")

        with st.sidebar:
            st.subheader("Settings")
            agent_name = st.selectbox(
                "Select Bedrock Model",
                [
                    "anthropic.claude-3-sonnet-20240229-v1:0",
                    "anthropic.claude-3-haiku-20240307-v1:0",
                    "amazon.titan-text-express-v1",
                    "ai21.j2-ultra-v1",
                    "cohere.command-text-v14:0"
                ]
            )
            st.divider()

        chat_container = st.container()
        with chat_container:
            if st.session_state.user_id in st.session_state.conversation_history:
                for message in st.session_state.conversation_history[st.session_state.user_id]:
                    role = message["role"]
                    content = message["content"]
                    st.chat_message(role).write(content)

        while st.session_state.waiting_for_response:
            try:
                while not st.session_state.response_queue.empty():
                    user_id, text_chunk, is_error = st.session_state.response_queue.get(timeout=0.1)
                    if is_error:
                        st.error(text_chunk)
                    else:
                        st.rerun()
            except queue.Empty:
                continue

        user_prompt = st.chat_input("Enter your prompt here", disabled=st.session_state.is_processing)

        if user_prompt:
            st.chat_message("user").write(user_prompt)
            st.session_state.is_processing = True

            with st.chat_message("assistant"):
                with st.status("Generating response...", expanded=True) as status:
                    st.write("Please wait while AWS Bedrock processes your request...")
                    st.session_state.waiting_for_response = True

                    thread = threading.Thread(
                        target=lambda: self.process_request(
                            user_prompt,
                            st.session_state.user_id,
                            st.session_state.session_id
                        )
                    )
                    add_script_run_ctx(thread)
                    thread.daemon = True
                    thread.start()

                    while thread.is_alive():
                        time.sleep(0.1)
                        if not st.session_state.waiting_for_response:
                            break

                    status.update(label="Response received!", state="complete", expanded=False)

            st.rerun()

    def run(self):
        """Main application flow"""
        self.chat_interface()

if __name__ == "__main__":
    app = BedrockChatApp()
    app.run()