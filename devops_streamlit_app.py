import json
import queue
import threading
import time
import uuid
import os
from datetime import datetime

import boto3
import streamlit as st
from streamlit.runtime.scriptrunner import get_script_run_ctx, add_script_run_ctx
from botocore.exceptions import BotoCoreError, ClientError

class BedrockChatApp:
    def __init__(self):
        self.initialize_state()
        self.configure_page()
        self.load_config()

    def load_config(self):
        """Load configuration from environment variables"""
        self.region = os.getenv('AWS_REGION', 'us-east-1')
        self.agent_alias_id = os.getenv('BEDROCK_AGENT_ALIAS_ID')
        self.agent_id = os.getenv('BEDROCK_AGENT_ID')

        if not all([self.agent_alias_id, self.agent_id]):
            raise ValueError("Required environment variables not set")

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

    def configure_page(self):
        """Configure Streamlit page settings"""
        st.set_page_config(
            page_title="AWS Bedrock Chat",
            page_icon="🤖",
            layout="wide"
        )

    def validate_input(self, prompt):
        """Validate user input"""
        if not prompt or len(prompt.strip()) == 0:
            return False
        if len(prompt) > 4000:  # Example max length
            return False
        return True

    def invoke_bedrock_model_with_streaming(self, prompt, user_id, session_id):
        """Invoke AWS Bedrock model with streaming response"""
        try:
            bedrock_client = boto3.client('bedrock-agent-runtime', region_name=self.region)
            
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

        except (BotoCoreError, ClientError) as e:
            error_msg = f"AWS API Error: {str(e)}"
            st.session_state.response_queue.put((user_id, error_msg, True))
            return None
        except Exception as e:
            error_msg = f"Unexpected error: {str(e)}"
            st.session_state.response_queue.put((user_id, error_msg, True))
            return None

    def process_request(self, prompt, user_id, session_id):
        """Process the user request and get a response from AWS Bedrock"""
        try:
            if not self.validate_input(prompt):
                raise ValueError("Invalid input")

            if user_id not in st.session_state.conversation_history:
                st.session_state.conversation_history[user_id] = []

            st.session_state.conversation_history[user_id].append({"role": "user", "content": prompt})
            st.session_state.waiting_for_response = True

            full_response = self.invoke_bedrock_model_with_streaming(prompt, user_id, session_id)
            
            if full_response:
                st.write(full_response)
                st.session_state.conversation_history[user_id].append({
                    "role": "assistant",
                    "content": full_response,
                    "timestamp": datetime.now().isoformat()
                })

        except Exception as e:
            st.error(f"Error processing request: {str(e)}")
        finally:
            st.session_state.is_processing = False
            st.session_state.waiting_for_response = False

    # ... rest of the class implementation remains the same ...

if __name__ == "__main__":
    try:
        app = BedrockChatApp()
        app.run()
    except Exception as e:
        st.error(f"Application error: {str(e)}")