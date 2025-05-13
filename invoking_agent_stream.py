import os
import uuid
import boto3
from botocore.exceptions import BotoCoreError, ClientError

def get_response(response):
    """Process and yield response chunks from Bedrock."""
    if response.get('completion'):
        for event_chunk in response['completion']:
            if 'chunk' in event_chunk and 'bytes' in event_chunk['chunk']:
                try:
                    chunk_text = event_chunk['chunk']['bytes'].decode('utf-8')
                    yield chunk_text
                except UnicodeDecodeError as e:
                    print(f"Error decoding response: {e}")

def get_bedrock_client():
    """Create and return configured Bedrock client."""
    region = os.getenv('AWS_REGION', 'us-east-1')
    return boto3.client('bedrock-agent-runtime', region_name=region)

def main():
    try:
        bedrock_client = get_bedrock_client()
        
        user_input = """
        Help me analyze the repo https://github.com/VinodKumarKP/python_project.git for any code issues.
        """

        # Get agent configuration from environment
        agent_alias_id = os.getenv('BEDROCK_AGENT_ALIAS_ID')
        agent_id = os.getenv('BEDROCK_AGENT_ID')
        
        if not agent_alias_id or not agent_id:
            raise ValueError("Missing required environment variables for agent configuration")

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
        print(f"AWS API error: {aws_error}")
    except Exception as e:
        print(f"Unexpected error: {e}")

if __name__ == "__main__":
    main()