import uuid
import os
import boto3
from botocore.exceptions import ClientError

def get_response(response):
    """Process and yield chunks from the Bedrock response"""
    if response.get('completion'):
        for event_chunk in response['completion']:
            if 'chunk' in event_chunk and 'bytes' in event_chunk['chunk']:
                try:
                    chunk_text = event_chunk['chunk']['bytes'].decode('utf-8')
                    yield chunk_text
                except UnicodeDecodeError as e:
                    print(f"Error decoding response: {e}")

def main():
    # Load configuration from environment variables
    region = os.getenv('AWS_REGION', 'us-east-1')
    agent_alias_id = os.getenv('BEDROCK_AGENT_ALIAS_ID')
    agent_id = os.getenv('BEDROCK_AGENT_ID')

    if not all([agent_alias_id, agent_id]):
        raise ValueError("Missing required environment variables")

    bedrock_client = boto3.client('bedrock-agent-runtime', region_name=region)

    user_input = """
    Help me analyze the repo https://github.com/VinodKumarKP/python_project.git for any code issues.
    """

    session_id = str(uuid.uuid4())

    try:
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

    except ClientError as e:
        print(f"AWS API error: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")

if __name__ == "__main__":
    main()