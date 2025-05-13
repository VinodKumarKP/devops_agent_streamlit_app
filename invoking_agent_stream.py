import uuid
import os
from typing import Generator
import boto3
from botocore.exceptions import BotoCoreError, ClientError

def validate_input(input_text: str) -> bool:
    """Validate user input."""
    if not input_text or len(input_text.strip()) == 0:
        raise ValueError("Input text cannot be empty")
    if len(input_text) > 1000:  # Set appropriate limit
        raise ValueError("Input text too long")
    return True

def get_response(response) -> Generator[str, None, None]:
    """Process and yield response chunks."""
    if response.get('completion'):
        for event_chunk in response['completion']:
            if 'chunk' in event_chunk and 'bytes' in event_chunk['chunk']:
                try:
                    chunk_text = event_chunk['chunk']['bytes'].decode('utf-8')
                    yield chunk_text
                except UnicodeDecodeError as e:
                    print(f"Error decoding response: {e}")

def main():
    try:
        # Get configuration from environment variables
        region = os.getenv('AWS_REGION', 'us-east-1')
        agent_id = os.getenv('AWS_AGENT_ID')
        agent_alias_id = os.getenv('AWS_AGENT_ALIAS_ID')

        if not all([agent_id, agent_alias_id]):
            raise ValueError("Missing required environment variables")

        bedrock_client = boto3.client('bedrock-agent-runtime', region_name=region)

        user_input = """
        Help me analyze the repo https://github.com/VinodKumarKP/python_project.git for any code issues.
        """

        # Validate input
        validate_input(user_input)

        session_id = str(uuid.uuid4())

        response = bedrock_client.invoke_agent(
            agentAliasId=agent_alias_id,
            agentId=agent_id,
            enableTrace=True,
            endSession=False,
            inputText=user_input,
            sessionId=session_id,
            streamingConfigurations={'streamFinalResponse': True}
        )

        for chunk in get_response(response):
            print(chunk, end='')

    except (BotoCoreError, ClientError) as aws_error:
        print(f"AWS Error: {aws_error}")
    except ValueError as ve:
        print(f"Validation Error: {ve}")
    except Exception as e:
        print(f"Unexpected error: {e}")

if __name__ == "__main__":
    main()