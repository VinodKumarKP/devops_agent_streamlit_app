import json
import queue
import threading
import time
import uuid
from datetime import datetime

import boto3
import streamlit as st
from streamlit.runtime.scriptrunner import get_script_run_ctx, add_script_run_ctx

from time import sleep


class BedrockChatApp:
    def __init__(self):
        # Initialize application state
        self.initialize_state()
        self.configure_page()

    def initialize_state(self):
        """Initialize all session state variables"""
        if "user_id" not in st.session_state:
            st.session_state.user_id = str(uuid.uuid4())

        if "conversation_history" not in st.session_state:
            st.session_state.conversation_history = {}

        # if "current_user" not in st.session_state:
        #     st.session_state.current_user = None

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

    def invoke_bedrock_model_with_streaming(self, prompt, user_id, session_id):
        """Invoke AWS Bedrock model with streaming response"""
        bedrock_client = boto3.client('bedrock-agent-runtime', region_name='us-east-1')

        try:
            response = bedrock_client.invoke_agent(
                agentAliasId='NW6OFIOLFM',
                agentId='UHPMS1A2QV',
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
                    # else:
                    #     st.session_state.response_queue.put((user_id, 'waiting for response', False))

            return full_response

        except Exception as e:
            error_msg = f"Error: {str(e)}"
            st.session_state.response_queue.put((user_id, error_msg, True))
            return None

    def process_request(self, prompt, user_id, session_id):
        """Process the user request and get a response from AWS Bedrock"""
        try:
            if user_id not in st.session_state.conversation_history:
                st.session_state.conversation_history[user_id] = []

            st.session_state.conversation_history[user_id].append({"role": "user", "content": prompt})
            st.session_state.waiting_for_response = True

            full_response = self.invoke_bedrock_model_with_streaming(prompt, user_id, session_id)
            st.write(full_response)

            if full_response:
                st.session_state.conversation_history[user_id].append({
                    "role": "assistant",
                    "content": full_response,
                    "timestamp": datetime.now().isoformat()
                })
                st.session_state.waiting_for_response = False


        except Exception as e:
            st.error(f"Error processing request: {str(e)}")
        finally:
            # Reset processing flag
            st.session_state.is_processing = False
            st.session_state.waiting_for_response = False

    def chat_interface(self):
        """Display chat interface"""
        st.title("AWS Bedrock Chat")

        # Sidebar for settings
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

                    if role == "user":
                        st.chat_message("user").write(content)
                    else:
                        st.chat_message("assistant").write(content)

        # Process any queued responses
        print(st.session_state.waiting_for_response, st.session_state.is_processing)
        while st.session_state.waiting_for_response:
            if not st.session_state.response_queue.empty():
                try:
                    # Process all available responses
                    while not st.session_state.response_queue.empty():
                        user_id, text_chunk, is_error = st.session_state.response_queue.get(block=False)

                        if is_error:
                            st.error(text_chunk)
                        else:
                            st.rerun()

                except queue.Empty:
                    print("Queue is empty")
                    pass

        user_prompt = st.chat_input("Enter your prompt here", disabled=st.session_state.is_processing)

        if user_prompt:
            # Display user message
            st.chat_message("user").write(user_prompt)
            st.session_state.is_processing = True

            with st.chat_message("assistant"):
                with st.status("Generating response...", expanded=True) as status:
                    st.write("Please wait while AWS Bedrock processes your request. Response may take a few minutes depending upon the number of files.")
                    st.session_state.waiting_for_response = True

                    time.sleep(1)
                    # Start new thread to process request
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

                    time.sleep(0.1)  # Small delay for thread to start
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