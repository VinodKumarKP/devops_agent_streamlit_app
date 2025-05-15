import os

import boto3
from botocore.config import Config


class AWSClientManager:
    """
    Manages AWS client connections for Bedrock services.
    """

    def __init__(self):
        """Initialize AWS client connections with appropriate configuration."""
        self.boto3_config = Config(read_timeout=1000)
        self.region = os.environ.get('AWS_REGION')
        if not self.region:
            raise EnvironmentError("AWS_REGION environment variable is not set")

        self.bedrock_client = boto3.client(
            'bedrock-agent-runtime',
            self.region,
            config=self.boto3_config
        )

        self.bedrock_agent_client = boto3.client(
            'bedrock-agent',
            self.region,
            config=self.boto3_config
        )
