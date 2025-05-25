import queue
import uuid

import streamlit as st


class SessionManager:
    """
    Manages Streamlit session state and user sessions.
    """

    def initialize_state(self):
        """Initialize all session state variables."""
        # User identification
        if "user_id" not in st.session_state:
            st.session_state.user_id = str(uuid.uuid4())

        if "session_id" not in st.session_state:
            st.session_state.session_id = str(uuid.uuid4())

        # Conversation tracking
        if "conversation_history" not in st.session_state:
            st.session_state.conversation_history = {}

        # Authentication status
        if "is_authenticated" not in st.session_state:
            st.session_state.is_authenticated = False

        # Processing status flags
        if "is_processing" not in st.session_state:
            st.session_state.is_processing = False

        if "waiting_for_response" not in st.session_state:
            st.session_state.waiting_for_response = False

        # User management
        if "user_database" not in st.session_state:
            st.session_state.user_database = {}

        # Response streaming
        if "response_queue" not in st.session_state:
            st.session_state.response_queue = queue.Queue()

        # Track the previously selected agent
        if 'previous_agent_key' not in st.session_state:
            st.session_state.previous_agent_key = None

        if 'placeholder' not in st.session_state:
            st.session_state.placeholder = None
