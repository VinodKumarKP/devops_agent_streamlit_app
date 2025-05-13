import json
import queue
import threading
import time
import uuid
import os
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

import boto3
import streamlit as st
from streamlit.runtime.scriptrunner import get_script_run_ctx, add_script_run_ctx

MAX_THREADS = 3
MAX_PROMPT_LENGTH = 1000

class BedrockChatApp:
    def __init__(self):
        self.initialize_state()
        self.configure_page()
        self.thread_pool = ThreadPoolExecutor(max_workers=MAX_THREADS)
        
        # Load configuration
        self.region = os.getenv('AWS_REGION', 'us-east-1')
        self.agent_alias_id = os.getenv('BEDROCK_AGENT_ALIAS_ID')
        self.agent_id = os.getenv('BEDROCK_AGENT_ID')

    def initialize_state(self):
        """Initialize all session state variables"""
        if "user_id" not in st.session_state:
            st.session_state.user_id = str(uuid.uuid4())
        if "conversation_history" not in st.session_state:
            st.session_state.conversation_history = {}
        if "is_authenticated" not in st.session_state:
            st.session_state.is_authenticated = False
        if "is_processing" not in st.session_state:
            st.session_state.is_processing = False
        if "user_database" not in st.session_state:
            st.session_state.user_database = {}
        if "response_queue" not in st.session_state:
            st.session_state.response_queue = queue.Queue()
        if "waiting_for_response" not in st.session_state:
            st.session_state.waiting_for_response = False
        if "session_id" not in st.session_state:
            st.session_state.session_id = str(uuid.uuid4())

    def validate_input(self, prompt):
        """Validate user input"""
        if not prompt or len(prompt.strip()) == 0:
            return False, "Empty prompt"
        if len(prompt) > MAX_PROMPT_LENGTH:
            return False, f"Prompt exceeds maximum length of {MAX_PROMPT_LENGTH}"
        return True, None

    def invoke_bedrock_model_with_streaming(self, prompt, user_id, session_id):
        """Invoke AWS Bedrock model with streaming response"""
        bedrock_client = boto3.client('bedrock-agent-runtime', region_name=self.region)

        try:
            response = bedrock_client.invoke_agent(
                agentAliasId=self.agent_alias_id,
                agentId=self.agent_id,
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
                        st.session_state.response_queue.put((user_id, text_chunk, False))

            return full_response

        except Exception as e:
            error_msg = f"Error: {str(e)}"
            st.session_state.response_queue.put((user_id, error_msg, True))
            return None

    # ... rest of the class implementation remains the same ...

    def __del__(self):
        """Cleanup thread pool on deletion"""
        self.thread_pool.shutdown(wait=True)