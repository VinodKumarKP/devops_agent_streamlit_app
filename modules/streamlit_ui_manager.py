import os
import queue
from typing import Dict

import streamlit as st

from modules.bedrock_agent_manager import BedrockAgentManager
from modules.constants import Constants


class StreamlitUIManager:
    """
    Manages the Streamlit user interface components.
    """

    def __init__(self, bedrock_agent_manager: BedrockAgentManager):
        self.bedrock_agent_manager = bedrock_agent_manager


    def configure_page(self):
        """Configure Streamlit page settings."""
        st.set_page_config(
            page_title="AWS Bedrock Chat",
            page_icon="🤖",
            layout="wide"
        )
        self.load_css()

    def display_header(self, title):
        hide_streamlit_style = """
        <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        </style>

        """
        st.markdown(hide_streamlit_style, unsafe_allow_html=True)
        st.set_option("client.toolbarMode", "viewer")

        st.markdown(f"""
        <div class="main-header">
            <h3>{title}</h3>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <style>
        header.stAppHeader {
            background-color: transparent;
        }
        section.stMain .block-container {
            margin: 0rem;
            padding-top: 0rem;
            padding-bottom: 0rem;
            padding-left: 0.5rem;
            padding-right: 0.5rem;
        }
        </style>""", unsafe_allow_html=True)

    def render_sidebar(self, config: Dict) -> tuple[str, str]:
        """
        Render the sidebar with settings and instructions.

        Args:
            config: Application configuration dictionary

        Returns:
            str: Selected agent name
        """
        with st.sidebar:
            st.subheader("Settings")
            agent_list = self.bedrock_agent_manager.get_agent_list()
            option_list = {values['name']:f"{key}:{agent}"
                           for agent in agent_list for key,values in config.items() if key in agent}

            agent_name = st.selectbox(
                "Select Bedrock Agent",
                options=[
                        option for option in option_list.keys()
                ]
            )

            agent_key = option_list[agent_name].split(":")[0]
            agent_name = option_list[agent_name].split(":")[1]

            st.divider()

            # st.text_area(
            #     "Instructions",
            #     value=config[agent_key]['instructions'],
            #     height=400,
            #     disabled=True
            # )
            st.markdown(config[agent_key]['instructions'], unsafe_allow_html=True)

        return agent_name,agent_key

    def render_chat_history(self, user_id: str, conversation_history: Dict):
        """
        Render the chat history for the current user.

        Args:
            user_id: Current user ID
            conversation_history: Dictionary containing all conversation histories
        """
        if user_id in conversation_history:
            for message in conversation_history[user_id]:
                role = message["role"]
                content = message["content"]

                if role == "user":
                    st.chat_message("user", avatar=Constants.USER_AVATAR).write(content)
                else:
                    st.chat_message("assistant", avatar = Constants.ASSISTANT_AVATAR).write(content)

    def process_response_queue(self):
        """Process any queued streaming responses."""
        while (st.session_state.waiting_for_response or
               not st.session_state.response_queue.empty()):
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
                    pass


    def load_css(self):
        """Load CSS styles for the application."""

        directory_name = os.path.dirname(__file__)
        config_path = os.path.join(os.path.dirname(directory_name), 'style.css')

        with open(config_path) as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)