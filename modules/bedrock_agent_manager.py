import subprocess
import time
from typing import Optional, Dict

import streamlit as st

from modules.aws_client_manager import AWSClientManager
from modules.mcp_client import MCPBedrockClient


class BedrockAgentManager:
    """
    Manages interactions with AWS Bedrock Agents.
    """

    def __init__(self, aws_clients: AWSClientManager, ):
        """
        Initialize the Bedrock Agent Manager.

        Args:
            aws_clients: Initialized AWS client manager
        """
        self.bedrock_client = aws_clients.bedrock_client
        self.bedrock_agent_client = aws_clients.bedrock_agent_client
        self.mcp_client = MCPBedrockClient(region_name=aws_clients.region)
        self.placeholder = None

    def get_agent_list(self):
        """
        Retrieve a list of available agents.

        Returns:
            list: List of agent dictionaries
        """
        response = self.bedrock_agent_client.list_agents()
        return [ agent['agentName'] for agent in response['agentSummaries']]

    def get_agent_id(self, agent_name: str) -> str:
        """
        Retrieve the agent ID for a given agent name.

        Args:
            agent_name: Name of the agent to find

        Returns:
            str: Agent ID

        Raises:
            ValueError: If the agent is not found
        """
        response = self.bedrock_agent_client.list_agents()

        agent_id = None
        if 'agentSummaries' in response:
            for agent in response['agentSummaries']:
                if agent_name in agent['agentName']:
                    agent_id = agent['agentId']
                    break

        if agent_id is None:
            raise ValueError(f"Agent {agent_name} not found")

        return agent_id

    def get_agent_alias_id(self, agent_id: str, agent_name: str) -> str:
        """
        Retrieve the alias ID for a given agent.

        Args:
            agent_id: ID of the agent
            agent_name: Name of the agent (for error reporting)

        Returns:
            str: Agent alias ID

        Raises:
            ValueError: If the agent alias is not found
        """
        response = self.bedrock_agent_client.list_agent_aliases(agentId=agent_id)

        alias_id = None
        if 'agentAliasSummaries' in response:
            for alias in response['agentAliasSummaries']:
                if 'latest' in alias['agentAliasName']:
                    alias_id = alias['agentAliasId']
                    break

        if alias_id is None:
            raise ValueError(f"Alias for agent {agent_name} not found")

        return alias_id

    def invoke_agent(
            self,
            prompt: str,
            user_id: str,
            session_id: str,
            agent_name: str,
            agent_type: str,
            agent_config: Optional[Dict] = None
    ) -> Optional[str]:
        """
        Invoke AWS Bedrock agent with streaming response.

        Args:
            prompt: User input prompt
            user_id: Unique user identifier
            session_id: Current session identifier
            agent_name: Name of the agent to invoke
            agent_type: Type of the agent (e.g., 'bedrock', 'mcp')
            agent_config: Optional configuration for the agent

        Returns:
            Optional[str]: Full response from the agent, or None on error
        """
        try:
            if agent_type == 'bedrock':
                agent_id = self.get_agent_id(agent_name)
                alias_agent_id = self.get_agent_alias_id(agent_id=agent_id, agent_name=agent_name)

                response = self.bedrock_client.invoke_agent(
                    agentAliasId=alias_agent_id,
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
                        text_chunk = ''
                        if "chunk" in event:
                            chunk = event["chunk"]
                            text_chunk = chunk.get("bytes").decode()

                        if text_chunk:
                            full_response += text_chunk
                            st.session_state.response_queue.put((user_id, text_chunk, False))

                return full_response
            else:
                self.mcp_client.set_command(self.which(agent_config.get('command')))
                self.mcp_client.set_server_script(agent_config.get('server_script', []))
                self.mcp_client.set_system_prompt(agent_config.get('system_prompt'))
                self.mcp_client.set_progress_callback(self.progress_callable)
                return self.mcp_client.process_mcp_response(prompt, user_id)

        except Exception as e:
            error_msg = f"Error invoking Bedrock agent: {str(e)}"
            st.error(error_msg)
            st.session_state.response_queue.put((user_id, error_msg, True))
            return None

    def which(self, program):
        try:
            result = subprocess.run(['which', program], capture_output=True, text=True, check=True)
            return result.stdout.strip()
        except subprocess.CalledProcessError:
            raise RuntimeError(f"'{program}' is not found in the system path.")

    def progress_callable(self, message: str):
        """
        Enhanced progress callback with improved animations and visual feedback.

        Args:
            message: Progress message to display
        """
        # Generate unique ID for this progress step
        step_id = f"progress_{hash(message) % 10000}"

        if st.session_state.placeholder is None:
            st.session_state.placeholder = st.empty()

        with st.session_state.placeholder:
            # Display the progress message
            st.markdown(f"""
                    <style>
                    .enhanced-progress-container {{
                        display: flex;
                        flex-direction: column;
                        align-items: center;
                        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                        margin: 20px 0;
                        position: relative;
                    }}

                    .progress-box-{step_id} {{
                        width: 520px;
                        height: 90px;
                        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                        border: 2px solid transparent;
                        border-radius: 16px;
                        display: flex;
                        align-items: center;
                        justify-content: space-between;
                        padding: 0 24px;
                        color: white;
                        font-weight: 600;
                        font-size: 16px;
                        text-align: left;
                        box-shadow: 
                            0 8px 32px rgba(102, 126, 234, 0.3),
                            0 4px 16px rgba(0, 0, 0, 0.2);
                        margin: 8px 0;
                        position: relative;
                        overflow: hidden;
                        animation: slideInScale-{step_id} 0.8s cubic-bezier(0.34, 1.56, 0.64, 1) forwards;
                        transform: translateY(20px) scale(0.9);
                        opacity: 0;
                    }}

                    .progress-box-{step_id}::before {{
                        content: '';
                        position: absolute;
                        top: 0;
                        left: -100%;
                        width: 100%;
                        height: 100%;
                        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.3), transparent);
                        animation: shimmer-{step_id} 2s ease-in-out infinite;
                    }}



                    .progress-spinner-{step_id} {{
                        width: 24px;
                        height: 24px;
                        border: 3px solid rgba(255,255,255,0.3);
                        border-top: 3px solid white;
                        border-radius: 50%;
                        animation: spin-{step_id} 1s linear infinite;
                    }}

                    .progress-line-{step_id} {{
                        width: 4px;
                        height: 0;
                        background: linear-gradient(180deg, #667eea, #764ba2);
                        position: relative;
                        border-radius: 2px;
                        animation: growLine-{step_id} 1s cubic-bezier(0.25, 0.46, 0.45, 0.94) 0.5s forwards;
                        box-shadow: 0 0 10px rgba(102, 126, 234, 0.5);
                        margin: 5px 0;
                    }}

                    .progress-line-{step_id}::before {{
                        content: '';
                        position: absolute;
                        top: 0;
                        left: 50%;
                        transform: translateX(-50%);
                        width: 8px;
                        height: 8px;
                        background: #667eea;
                        border-radius: 50%;
                        animation: pulse-{step_id} 2s ease-in-out infinite;
                        box-shadow: 0 0 15px rgba(102, 126, 234, 0.8);
                    }}

                    .progress-line-{step_id}::after {{
                        content: '';
                        position: absolute;
                        bottom: -8px;
                        left: 50%;
                        transform: translateX(-50%);
                        border: 8px solid transparent;
                        border-top: 12px solid #764ba2;
                        filter: drop-shadow(0 2px 4px rgba(0,0,0,0.3));
                        animation: arrowFade-{step_id} 1s ease-out 1.2s forwards;
                        opacity: 0;
                    }}

                    .progress-bar-container-{step_id} {{
                        width: 520px;
                        margin: 15px 0;
                    }}

                    .progress-bar-{step_id} {{
                        width: 100%;
                        height: 6px;
                        background-color: rgba(255,255,255,0.1);
                        border-radius: 3px;
                        overflow: hidden;
                        position: relative;
                    }}

                    .progress-fill-{step_id} {{
                        height: 100%;
                        background: linear-gradient(90deg, #667eea, #764ba2);
                        border-radius: 3px;
                        animation: progressFill-{step_id} 2s ease-out forwards;
                        transform-origin: left;
                        transform: scaleX(0);
                    }}

                    .progress-fill-{step_id}::after {{
                        content: '';
                        position: absolute;
                        top: 0;
                        right: 0;
                        width: 20px;
                        height: 100%;
                        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.6), transparent);
                        animation: progressShine-{step_id} 1.5s ease-in-out infinite;
                    }}

                    /* Keyframe Animations */
                    @keyframes slideInScale-{step_id} {{
                        to {{
                            transform: translateY(0) scale(1);
                            opacity: 1;
                        }}
                    }}

                    @keyframes shimmer-{step_id} {{
                        0% {{ left: -100%; }}
                        50% {{ left: 100%; }}
                        100% {{ left: 100%; }}
                    }}



                    @keyframes growLine-{step_id} {{
                        to {{ height: 60px; }}
                    }}

                    @keyframes pulse-{step_id} {{
                        0%, 100% {{ 
                            transform: translateX(-50%) scale(1);
                            opacity: 1;
                        }}
                        50% {{ 
                            transform: translateX(-50%) scale(1.5);
                            opacity: 0.7;
                        }}
                    }}

                    @keyframes arrowFade-{step_id} {{
                        to {{ opacity: 1; }}
                    }}

                    @keyframes spin-{step_id} {{
                        0% {{ transform: rotate(0deg); }}
                        100% {{ transform: rotate(360deg); }}
                    }}

                    @keyframes progressFill-{step_id} {{
                        to {{ transform: scaleX(1); }}
                    }}

                    @keyframes progressShine-{step_id} {{
                        0% {{ transform: translateX(-20px); }}
                        100% {{ transform: translateX(20px); }}
                    }}
                    </style>

                    <div class="enhanced-progress-container">
                        <div class="progress-box-{step_id}">
                            <span>🔄 {message}</span>
                            <div class="progress-spinner-{step_id}"></div>
                        </div>
                        <div class="progress-line-{step_id}"></div>
                        <div class="progress-bar-container-{step_id}">
                            <div class="progress-bar-{step_id}">
                                <div class="progress-fill-{step_id}"></div>
                            </div>
                        </div>
                    </div>
                """, unsafe_allow_html=True)

            # Shorter delay for better responsiveness
            time.sleep(1)