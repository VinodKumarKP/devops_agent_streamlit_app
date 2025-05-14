```python
import uuid
import os
import logging
import boto3

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_response(response):
    """Safely extract response chunks"""
    try:
        if response.get('completion'):
            for event_chunk in response['completion']:
                if 'chunk' in event_chunk and 'bytes' in event_chunk['chunk']:
                    chunk_text = event_chunk['chunk']['bytes'].decode('utf-8')
                    yield chunk_text
    except Exception as e:
        logger.error(f"Response parsing error: {e}")

def main():
    try:
        bedrock_client = boto3.client('bedrock-agent-runtime')
        
        user_input = input("Enter your analysis request: ")
        
        if not user_input or len(user_input) > 1000:
            raise ValueError("Invalid input length")

        session_id = str(uuid.uuid4())
        
        response = bedrock_client.invoke_agent(
            agentAliasId=os.getenv('BEDROCK_AGENT_ALIAS_ID', 'default_alias'),
            agentId=os.getenv('BEDROCK_AGENT_ID', 'default_agent'),
            enableTrace=False,
            endSession=False,
            inputText=user_input,
            sessionId=session_id,
            streamingConfigurations={'streamFinalResponse': True}
        )

        for chunk in get_response(response):
            print(chunk, end='')

    except Exception as e:
        logger.error(f"Agent invocation error: {e}")

if __name__ == "__main__":
    main()
```