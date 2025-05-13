import uuid

import boto3


def get_response(response):
    if response.get('completion'):
        for event_chunk in response['completion']:
            if 'chunk' in event_chunk and 'bytes' in event_chunk['chunk']:
                chunk_text = event_chunk['chunk']['bytes'].decode('utf-8')
                yield chunk_text

bedrock_client = boto3.client('bedrock-agent-runtime', region_name='us-east-1')

user_input = """
Help me analyze the repo https://github.com/VinodKumarKP/python_project.git for any code issues.
"""

sessionId=str(uuid.uuid4())

response = bedrock_client.invoke_agent(
                agentAliasId='G01L7KIZP3',
                agentId='UHPMS1A2QV',
                enableTrace=True,
                endSession=False,
                inputText=user_input,
                sessionId=sessionId,
                streamingConfigurations={'streamFinalResponse': True}
            )

for chunk in get_response(response):
    print(chunk, end='')