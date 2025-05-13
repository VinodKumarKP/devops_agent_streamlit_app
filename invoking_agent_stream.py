import uuid
import os
import boto3
from botocore.exceptions import BotoCoreError, ClientError
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_response(response):
    """Process and yield response chunks from Bedrock."""
    try:
        if response.get('completion'):
            for event_chunk in response['completion']:
                if 'chunk' in event_chunk and 'bytes' in event_chunk['chunk']:
                    chunk_text = event_chunk['chunk']['bytes'].decode('utf-8')
                    yield chunk_text
    except Exception as e:
        logger.error(f"Error processing response: {str(e)}")
        raise

def main():
    try:
        # Get configuration from environment variables
        region = os.getenv('AWS_REGION', 'us-east-1')
        agent_id = os.getenv('BEDROCK_AGENT_ID')
        agent_alias_id = os.getenv('BEDROCK_AGENT_ALIAS_ID')

        if not all([agent_id, agent_alias_id]):
            raise ValueError("Missing required environment variables")

        bedrock_client = boto3.client('bedrock-agent-runtime', region_name=region)

        user_input = """
        Help me analyze the repo https://github.com/VinodKumarKP/python_project.git for any code issues.
        """

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
        logger.error(f"AWS Error: {str(aws_error)}")
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")

if __name__ == "__main__":
    main()