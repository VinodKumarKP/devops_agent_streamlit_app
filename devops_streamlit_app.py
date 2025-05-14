```python
import json
import queue
import threading
import time
import uuid
from datetime import datetime
import os
import logging

import boto3
import streamlit as st
from streamlit.runtime.scriptrunner import get_script_run_ctx, add_script_run_ctx

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

class BedrockChatApp:
    def __init__(self):
        # Initialize application state with enhanced security
        self.initialize_state()
        self.configure_page()
        self.configure_aws_credentials()

    def configure_aws_credentials(self):
        """Securely configure AWS credentials"""
        try:
            # Use environment variables or AWS IAM roles
            boto3.setup_default_session(
                region_name=os.getenv('AWS_REGION', 'us-east-1')
            )
        except Exception as e:
            logger.error(f"AWS Credential Configuration Error: {e}")
            st.error("Unable to configure AWS credentials. Please check your configuration.")

    def initialize_state(self):
        """Initialize all session state variables with enhanced security"""
        # Set maximum conversation history size
        MAX_HISTORY_SIZE = 50

        if "user_id" not in st.session_state:
            st.session_state.user_id = str(uuid.uuid4())

        if "conversation_history" not in st.session_state:
            st.session_state.conversation_history = {}

        if "is_authenticated" not in st.session_state:
            st.session_state.is_authenticated = False

        if "is_processing" not in st.session_state:
            st.session_state.is_processing = False

        if "response_queue" not in st.session_state:
            st.session_state.response_queue = queue.Queue(maxsize=100)

        if "waiting_for_response" not in st.session_state:
            st.session_state.waiting_for_response = False

        if "session_id" not in st.session_state:
            st.session_state.session_id = str(uuid.uuid4())

    def validate_input(self, prompt):
        """Validate and sanitize user input"""
        if not prompt or len(prompt) > 1000:
            raise ValueError("Invalid input: Prompt must be between 1-1000 characters")
        
        # Add additional sanitization as needed
        sanitized_prompt = prompt.strip()
        return sanitized_prompt

    def invoke_bedrock_model_with_streaming(self, prompt, user_id, session_id):
        """Invoke AWS Bedrock model with enhanced error handling and streaming response"""
        try:
            bedrock_client = boto3.client('bedrock-agent-runtime')
            
            # Use environment variables or secure configuration management
            agent_alias_id = os.getenv('BEDROCK_AGENT_ALIAS_ID', 'default_alias')
            agent_id = os.getenv('BEDROCK_AGENT_ID', 'default_agent')

            response = bedrock_client.invoke_agent(
                agentAliasId=agent_alias_id,
                agentId=agent_id,
                enableTrace=False,
                endSession=False,
                inputText=prompt,
                sessionId=session_id,
                streamingConfigurations={'streamFinalResponse': True}
            )

            full_response = ''
            if response.get('completion'):
                for event in response['completion']:
                    text_chunk = event.get('chunk', {}).get('bytes', b'').decode()
                    if text_chunk:
                        full_response += text_chunk
                        st.session_state.response_queue.put((user_id, text_chunk, False))

            return full_response

        except Exception as e:
            error_msg = f"Bedrock Invocation Error: {str(e)}"
            logger.error(error_msg)
            st.session_state.response_queue.put((user_id, error_msg, True))
            return None

    def process_request(self, prompt, user_id, session_id):
        """Process the user request with enhanced error handling"""
        try:
            # Validate input
            sanitized_prompt = self.validate_input(prompt)

            if user_id not in st.session_state.conversation_history:
                st.session_state.conversation_history[user_id] = []

            # Limit conversation history
            if len(st.session_state.conversation_history[user_id]) >= 50:
                st.session_state.conversation_history[user_id] = st.session_state.conversation_history[user_id][-50:]

            st.session_state.conversation_history[user_id].append({"role": "user", "content": sanitized_prompt})
            st.session_state.waiting_for_response = True

            full_response = self.invoke_bedrock_model_with_streaming(sanitized_prompt, user_id, session_id)

            if full_response:
                st.session_state.conversation_history[user_id].append({
                    "role": "assistant",
                    "content": full_response,
                    "timestamp": datetime.now().isoformat()
                })
                st.session_state.waiting_for_response = False

        except ValueError as ve:
            st.error(str(ve))
        except Exception as e:
            logger.error(f"Request Processing Error: {str(e)}")
            st.error("An unexpected error occurred. Please try again.")
        finally:
            st.session_state.is_processing = False
            st.session_state.waiting_for_response = False

    # Rest of the code remains similar with minor improvements in error handling and logging
```