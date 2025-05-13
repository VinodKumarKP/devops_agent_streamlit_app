import json
import queue
import threading
import time
import uuid
import os
import logging
from datetime import datetime

import boto3
import streamlit as st
from streamlit.runtime.scriptrunner import get_script_run_ctx, add_script_run_ctx

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BedrockChatApp:
    def __init__(self):
        self.initialize_state()
        self.configure_page()
        self.load_config()

    def load_config(self):
        """Load configuration from environment variables"""
        self.aws_region = os.getenv('AWS_REGION', 'us-east-1')
        self.agent_id = os.getenv('BEDROCK_AGENT_ID')
        self.agent_alias_id = os.getenv('BEDROCK_AGENT_ALIAS_ID')

        if not all([self.agent_id, self.agent_alias_id]):
            raise ValueError("Missing required environment variables")

    def initialize_state(self):
        """Initialize all session state variables"""
        state_vars = {
            "user_id": str(uuid.uuid4()),
            "conversation_history": {},
            "is_authenticated": False,
            "is_processing": False,
            "user_database": {},
            "response_queue": queue.Queue(),
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

    def validate_input(self, prompt):
        """Validate user input"""
        if not prompt or len(prompt.strip()) == 0:
            return False
        if len(prompt) > 4000:  # Example limit
            return False
        return True

    def invoke_bedrock_model_with_streaming(self, prompt, user_id, session_id):
        """Invoke AWS Bedrock model with streaming response"""
        if not self.validate_input(prompt):
            raise ValueError("Invalid input")

        bedrock_client = boto3.client('bedrock-agent-runtime', region_name=self.aws_region)

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
            logger.error(f"Error in Bedrock invocation: {str(e)}")
            error_msg = f"Error: {str(e)}"
            st.session_state.response_queue.put((user_id, error_msg, True))
            return None

    # ... [rest of the class implementation remains the same] ...

    def run(self):
        """Main application flow"""
        try:
            self.chat_interface()
        except Exception as e:
            logger.error(f"Application error: {str(e)}")
            st.error("An unexpected error occurred. Please try again later.")

if __name__ == "__main__":
    try:
        app = BedrockChatApp()
        app.run()
    except Exception as e:
        logger.error(f"Startup error: {str(e)}")
        st.error("Failed to start the application. Please check the configuration.")