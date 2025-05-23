# Configure logging
import asyncio
import json
import logging
import os
from typing import Dict, Any, List

import boto3
from mcp import StdioServerParameters, stdio_client, ClientSession

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MCPBedrockClient:
    def __init__(self, region_name: str = 'us-east-1'):
        """Initialize Bedrock client"""
        self.mcp_initialized = False
        self.session_context = None
        self.write = None
        self.read = None
        self.stdio_context = None
        self.bedrock_client = boto3.client(
            service_name='bedrock-runtime',
            region_name=region_name
        )
        self.model_id = 'us.anthropic.claude-3-5-sonnet-20241022-v2:0'
        self.mcp_session = None
        self.available_tools = {}
        self.main_loop = None
        self.command = None
        self.server_script = None
        self.system_prompt = None

    def set_command(self, command: str):
        """Set the command to be used for the MCP server"""
        if command is None:
            raise ValueError("Command not set. Please set the command before initializing.")

        if not os.path.exists(command):
            raise ValueError(f"Command {command} does not exist.")
        self.command = command

    def set_server_script(self, server_script: List[str]):
        """Set the server script to be used for the MCP server"""
        if server_script is None:
            raise ValueError("Server script not set. Please set the server script before initializing.")

        for script in server_script:
            if not os.path.exists(script):
                raise ValueError(f"Server script {script} does not exist.")

        self.server_script = server_script

    def set_system_prompt(self, system_prompt: str):
        """Set the system prompt to be used for the MCP server"""
        if system_prompt is None:
            raise ValueError("System prompt not set. Please set the system prompt before initializing.")

        self.system_prompt = system_prompt


    async def initialize_mcp_session(self):
        """Initialize MCP session once and reuse"""
        try:
            logger.info("Initializing MCP session...")

            if self.server_script is None:
                raise ValueError("Server script not set. Please set the server script before initializing.")

            if self.command is None:
                raise ValueError("Command not set. Please set the command before initializing.")

            # Store reference to current event loop
            self.main_loop = asyncio.get_event_loop()

            server_params = StdioServerParameters(
                command=self.command,
                args=self.server_script
            )

            # Store the client context for reuse
            self.stdio_context = stdio_client(server_params)
            self.read, self.write = await self.stdio_context.__aenter__()

            self.session_context = ClientSession(self.read, self.write)
            self.mcp_session = await self.session_context.__aenter__()

            await self.mcp_session.initialize()

            # Load available tools once
            tools_response = await self.mcp_session.list_tools()
            for tool in tools_response.tools:
                self.available_tools[tool.name] = {
                    'description': tool.description,
                    'schema': tool.inputSchema
                }
                logger.info(f"Loaded tool: {tool.name}")

            return True

        except Exception as e:
            logger.error(f"Failed to initialize MCP session: {e}")
            return False

    async def cleanup_mcp_session(self):
        """Cleanup MCP session"""
        try:
            if self.session_context:
                await self.session_context.__aexit__(None, None, None)
            if self.stdio_context:
                await self.stdio_context.__aexit__(None, None, None)
        except Exception as e:
            # logger.error(f"Error cleaning up MCP session: {e}")
            pass

    async def execute_mcp_tool(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        """Execute a tool via MCP using existing session"""
        try:
            logger.info(f"Executing MCP tool: {tool_name} with args: {arguments}")

            result = await self.mcp_session.call_tool(tool_name, arguments)

            if result.content:
                text_content = []
                for content in result.content:
                    if hasattr(content, 'text'):
                        text_content.append(content.text)
                return '\n'.join(text_content)

            return "Tool executed successfully"

        except Exception as e:
            logger.error(f"Error executing tool {tool_name}: {e}")
            return f"Error: {str(e)}"

    def get_bedrock_tools_config(self) -> Dict[str, Any]:
        """Convert MCP tools to Bedrock format"""
        bedrock_tools = []

        for tool_name, tool_info in self.available_tools.items():
            bedrock_tool = {
                "name": tool_name,
                "description": tool_info['description'],
                "input_schema": tool_info['schema']
            }
            bedrock_tools.append(bedrock_tool)

        return {
            "tools": bedrock_tools,
            "tool_choice": {"type": "auto"}
        }

    async def query_bedrock_with_mcp(self, user_message: str) -> str:
        """Query Bedrock using existing MCP session"""
        try:

            messages = [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": user_message}]
                }
            ]

            body = {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 1000,
                "system": self.system_prompt,
                "messages": messages,
                **self.get_bedrock_tools_config()
            }

            logger.info(f"Sending request to Bedrock: {user_message}")

            response = self.bedrock_client.invoke_model(
                modelId=self.model_id,
                body=json.dumps(body)
            )

            response_body = json.loads(response['body'].read())
            return await self.process_response_with_mcp(response_body, messages)

        except Exception as e:
            logger.error(f"Error in Bedrock query: {e}")
            return f"Error: {str(e)}"

    async def process_response_with_mcp(self, response_body: Dict[str, Any],
                                        conversation_history: List[Dict]) -> str:
        """Process Bedrock response using existing MCP session"""
        max_iterations = 10
        iteration_count = 0
        current_response = response_body

        while iteration_count < max_iterations:
            iteration_count += 1
            content = current_response.get('content', [])

            tool_calls = []
            text_response = ""

            for item in content:
                if item.get('type') == 'text':
                    text_response += item.get('text', '')
                elif item.get('type') == 'tool_use':
                    tool_calls.append(item)

            if not tool_calls:
                if not text_response.strip() and iteration_count > 1:
                    return "Task completed successfully using MCP tools."
                return text_response

            logger.info(f"Iteration {iteration_count}: Executing {len(tool_calls)} MCP tools")

            conversation_history.append({
                "role": "assistant",
                "content": content
            })

            # Execute MCP tools using existing session
            tool_results = []
            for tool_call in tool_calls:
                tool_name = tool_call.get('name')
                tool_input = tool_call.get('input', {})
                tool_use_id = tool_call.get('id')

                # Execute via existing MCP session
                mcp_result = await self.execute_mcp_tool(tool_name, tool_input)

                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": tool_use_id,
                    "content": [{"type": "text", "text": mcp_result}]
                })

            conversation_history.append({
                "role": "user",
                "content": tool_results
            })

            # Continue conversation
            body = {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 2000,
                "system": self.system_prompt,
                "messages": conversation_history,
                **self.get_bedrock_tools_config()
            }

            try:
                next_response = self.bedrock_client.invoke_model(
                    modelId=self.model_id,
                    body=json.dumps(body)
                )
                current_response = json.loads(next_response['body'].read())

            except Exception as e:
                logger.error(f"Error in iteration {iteration_count}: {e}")
                return f"MCP tools executed, but error in response: {str(e)}"

        return "Maximum iterations reached."

    async def _handle_mcp_request(self, prompt: str, user_id: str) -> str:
        """Handle MCP-enhanced requests"""
        try:
            # Initialize MCP session if not already done
            if not self.mcp_initialized:
                success = await self.initialize_mcp_session()
                if not success:
                    return "Error: Could not initialize MCP tools"

            # Query with MCP support
            response = await self.query_bedrock_with_mcp(prompt)
            return response

        except Exception as e:
            error_msg = f"Error in MCP request: {str(e)}"
            logger.error(error_msg)
            return error_msg

    def process_mcp_response(self, prompt, user_id):
        import asyncio
        try:
            # If we're already in the main async context, run directly
            if self.main_loop and self.main_loop.is_running():
                # We're in an async context, but need to handle this properly
                # Create a task that can be awaited

                try:
                    # Try to run directly if nest_asyncio is working
                    return asyncio.run(self._handle_mcp_request(prompt, user_id))
                except RuntimeError:
                    # Use a thread to avoid event loop conflicts
                    import threading
                    result = [None]
                    exception = [None]

                    def run_async():
                        try:
                            loop = asyncio.new_event_loop()
                            asyncio.set_event_loop(loop)
                            result[0] = loop.run_until_complete(
                                self._handle_mcp_request(prompt, user_id)
                            )
                        except Exception as e:
                            exception[0] = e
                        finally:
                            loop.close()

                    thread = threading.Thread(target=run_async)
                    thread.start()
                    thread.join(timeout=300)

                    if exception[0]:
                        raise exception[0]
                    return result[0]
            else:
                # No main loop, run normally
                return asyncio.run(self._handle_mcp_request(prompt, user_id))
        except Exception as e:
            logger.error(f"Error in invoke_agent: {e}")
            return f"Error: {str(e)}"
        finally:
            # Cleanup MCP session
            asyncio.run(self.cleanup_mcp_session())

    def __del__(self):
        """Close the MCP session"""
        if self.mcp_session:
            asyncio.run(self.cleanup_mcp_session())
            self.mcp_session = None
            self.mcp_initialized = False
            logger.info("MCP session closed.")

    async def close(self):
        """Close the MCP session"""
        logger.info("Closing MCP session...")