from typing import Optional

import streamlit as st

from modules.aws_client_manager import AWSClientManager


class BedrockAgentManager:
    """
    Manages interactions with AWS Bedrock Agents.
    """

    def __init__(self, aws_clients: AWSClientManager):
        """
        Initialize the Bedrock Agent Manager.

        Args:
            aws_clients: Initialized AWS client manager
        """
        self.bedrock_client = aws_clients.bedrock_client
        self.bedrock_agent_client = aws_clients.bedrock_agent_client

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
                if 'devops-code-remediation-agent-dev' in agent['agentName']:
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
            agent_name: str
    ) -> Optional[str]:
        """
        Invoke AWS Bedrock agent with streaming response.

        Args:
            prompt: User input prompt
            user_id: Unique user identifier
            session_id: Current session identifier
            agent_name: Name of the agent to invoke

        Returns:
            Optional[str]: Full response from the agent, or None on error
        """
        try:
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

        except Exception as e:
            error_msg = f"Error invoking Bedrock agent: {str(e)}"
            st.error(error_msg)
            st.session_state.response_queue.put((user_id, error_msg, True))
            return None
