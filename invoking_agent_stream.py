import json
import os
import uuid

import boto3


def get_response(response):
    if response.get('completion'):
        for event_chunk in response['completion']:
            if 'chunk' in event_chunk and 'bytes' in event_chunk['chunk']:
                chunk_text = event_chunk['chunk']['bytes'].decode('utf-8')
                yield chunk_text

os.environ['AWS_REGION'] = 'us-east-2'

bedrock_client = boto3.client('bedrock-agent-runtime', os.environ['AWS_REGION'])
bedrock_agent_client = boto3.client('bedrock-agent', os.environ['AWS_REGION'])

user_input = """
Help me analyze the repo https://github.com/VinodKumarKP/python_project.git for any code issues.
"""

response = bedrock_agent_client.list_agents()
# print(response)

agentId = None
if 'agentSummaries' in response:
    for agent in response['agentSummaries']:
        if 'devops-code-remediation-agent-dev' in agent['agentName']:
            print(f"Agent ID: {agent['agentId']}")
            agentId = agent['agentId']


response = bedrock_agent_client.list_agent_aliases(agentId=agentId)
# print(response['agentAliasSummaries'])
aliasId = ''
if 'agentAliasSummaries' in response:
    for alias in response['agentAliasSummaries']:
        if 'latest' in alias['agentAliasName']:
            print(f"Alias ID: {alias['agentAliasId']}")
            aliasId = alias['agentAliasId']



sessionId=str(uuid.uuid4())
#
response = bedrock_client.invoke_agent(
                agentAliasId=aliasId,
                agentId=agentId,
                enableTrace=True,
                endSession=False,
                inputText=user_input,
                sessionId=sessionId,
                streamingConfigurations={'streamFinalResponse': True}
            )
#
for chunk in get_response(response):
    print(chunk, end='')