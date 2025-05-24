"""
AWS Bedrock Chat Application

This module provides a Streamlit-based chat interface for interacting with AWS Bedrock agents.
It handles user authentication, chat history management, and streaming responses from
AWS Bedrock models.

Author: Vinod Kumar KP
Date: May 14, 2025
"""
import asyncio
import threading
import time
from datetime import datetime

import streamlit as st
from streamlit.runtime.scriptrunner import add_script_run_ctx

from modules.aws_client_manager import AWSClientManager
from modules.bedrock_agent_manager import BedrockAgentManager
from modules.config_manager import ConfigManager
from modules.constants import Constants
from modules.session_manager import SessionManager
from modules.streamlit_ui_manager import StreamlitUIManager


class BedrockChatApp:
    """
    Main application class for the AWS Bedrock Chat application.
    """

    def __init__(self):
        """Initialize the chat application components."""
        # Initialize managers
        self.config_manager = ConfigManager()
        self.session_manager = SessionManager()
        self.aws_clients = AWSClientManager()
        self.agent_manager = BedrockAgentManager(self.aws_clients)
        self.ui_manager = StreamlitUIManager(self.agent_manager)

        # Set up application state
        self.session_manager.initialize_state()
        self.ui_manager.configure_page()

    def process_request(self,
                        prompt: str,
                        user_id: str,
                        session_id: str,
                        agent_name: str,
                        agent_type: str,
                        agent_config: dict = None):
        """
        Process the user request and get a response from AWS Bedrock.

        Args:
            prompt: User input prompt
            user_id: Unique user identifier
            session_id: Current session identifier
            agent_name: Name of the agent to invoke
            agent_type: Type of the agent (e.g., 'mcp', 'llm')
            agent_config: Optional configuration for the agent
        """
        try:
            # Initialize conversation history for this user if needed
            if user_id not in st.session_state.conversation_history:
                st.session_state.conversation_history[user_id] = []

            # Add user message to history
            st.session_state.conversation_history[user_id].append({
                "role": "user",
                "content": prompt
            })

            st.session_state.waiting_for_response = True

            conversation_history = ''
            if agent_type == 'mcp' and user_id in st.session_state.conversation_history and \
                len(st.session_state.conversation_history[user_id]) > 0:
                for message in reversed(st.session_state.conversation_history[user_id]):
                    if message['role'] == 'assistant':
                        conversation_history += message['content'] + "\n"
                        break
                prompt = conversation_history + "\n" + prompt


            # Get response from Bedrock
            full_response = self.agent_manager.invoke_agent(
                prompt,
                user_id,
                session_id,
                agent_name,
                agent_type,
                agent_config
            )

            # Log the full response
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
            # Reset processing flags
            st.session_state.is_processing = False
            st.session_state.waiting_for_response = False

    def chat_interface(self):
        """Display and manage the chat interface."""
        # Render sidebar and get selected agent
        agent_name,agent_key, agent_type = self.ui_manager.render_sidebar(self.config_manager.config)

        # Set app title
        self.ui_manager.display_header(self.config_manager.config[agent_key]['name'])

        # Display chat container with history
        chat_container = st.container()
        with chat_container:
            self.ui_manager.render_chat_history(
                st.session_state.user_id,
                st.session_state.conversation_history
            )

        # Process any streaming responses in queue
        self.ui_manager.process_response_queue()

        # Handle user input
        user_prompt = st.chat_input(
            "Enter your prompt here",
            disabled=st.session_state.is_processing
        )

        if user_prompt:
            # Display user message
            st.chat_message("user", avatar=Constants.USER_AVATAR).write(user_prompt)
            st.session_state.is_processing = True

            with st.chat_message("assistant", avatar=Constants.ASSISTANT_AVATAR):
                with st.status("Generating response...", expanded=True) as status:
                    st.write("Please wait while AWS Bedrock processes your request. "
                             "Response may take a few minutes depending upon the number of files.")
                    st.session_state.waiting_for_response = True

                    # Start new thread to process request
                    thread = threading.Thread(
                        target=lambda: self.process_request(
                            user_prompt,
                            st.session_state.user_id,
                            st.session_state.session_id,
                            agent_name,
                            agent_type,
                            self.config_manager.config[agent_key]
                        )
                    )
                    add_script_run_ctx(thread)
                    thread.daemon = True
                    thread.start()

                    # Wait for thread to complete
                    time.sleep(0.1)  # Small delay for thread to start
                    while thread.is_alive():
                        time.sleep(0.1)
                        if not st.session_state.waiting_for_response:
                            break

                    status.update(label="Response received!", state="complete", expanded=False)

            st.rerun()

    async def run(self):
        """Run the main application flow."""
        self.chat_interface()


if __name__ == "__main__":
    app = BedrockChatApp()
    asyncio.run(app.run())
