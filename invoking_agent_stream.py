import uuid
import os
import boto3
from botocore.exceptions import BotoCoreError, ClientError

def get_response(response):
    try:
        if response.get('completion'):
            for event_chunk in response['completion']:
                if 'chunk' in event_chunk and 'bytes' in event_chunk['chunk']:
                    chunk_text = event_chunk['chunk']['bytes'].decode('utf-8')
                    yield chunk_text
    except Exception as e:
        print(f"Error processing response: {str(e)}")
        yield None

# Load configuration from environment variables
REGION = os.getenv('AWS_REGION', 'us-east-1')
AGENT_ALIAS_ID = os.getenv('BEDROCK_AGENT_ALIAS_ID')
AGENT_ID = os.getenv('BEDROCK_AGENT_ID')

if not all([AGENT_ALIAS_ID, AGENT_ID]):
    raise ValueError("Required environment variables not set")

try:
    bedrock_client = boto3.client('bedrock-agent-runtime', region_name=REGION)
except Exception as e:
    print(f"Error initializing Bedrock client: {str(e)}")
    exit(1)

user_input = """
Help me analyze the repo https://github.com/VinodKumarKP/python_project.git for any code issues.
"""

sessionId = str(uuid.uuid4())

try:
    response = bedrock_client.invoke_agent(
        agentAliasId=AGENT_ALIAS_ID,
        agentId=AGENT_ID,
        enableTrace=True,
        endSession=False,
        inputText=user_input,
        sessionId=sessionId,
        streamingConfigurations={'streamFinalResponse': True}
    )

    for chunk in get_response(response):
        if chunk:
            print(chunk, end='')

except (BotoCoreError, ClientError) as e:
    print(f"AWS API error: {str(e)}")
except Exception as e:
    print(f"Unexpected error: {str(e)}")