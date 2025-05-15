import queue
from typing import Dict

import streamlit as st


class StreamlitUIManager:
    """
    Manages the Streamlit user interface components.
    """

    def configure_page(self):
        """Configure Streamlit page settings."""
        st.set_page_config(
            page_title="AWS Bedrock Chat",
            page_icon="🤖",
            layout="wide"
        )

    def render_sidebar(self, config: Dict) -> str:
        """
        Render the sidebar with settings and instructions.

        Args:
            config: Application configuration dictionary

        Returns:
            str: Selected agent name
        """
        with st.sidebar:
            st.subheader("Settings")
            agent_name = st.selectbox(
                "Select Bedrock Agent",
                options=[agent for agent in config.keys()]
            )

            st.divider()

            st.text_area(
                "Instructions",
                value=config[agent_name]['instructions'],
                height=400,
                disabled=True
            )

        return agent_name

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
                    st.chat_message("user").write(content)
                else:
                    st.chat_message("assistant").write(content)

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
