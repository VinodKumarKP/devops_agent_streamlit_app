import os

import boto3
from botocore.config import Config


class AWSClientManager:
    """
    Manages AWS client connections for Bedrock services.
    """

    def __init__(self):
        """
        Initialize AWS client connections with appropriate configuration.

        Attributes:
            boto3_config (Config): Configuration object for boto3 with a custom read timeout.
            region (str): AWS region retrieved from the 'AWS_REGION' environment variable.
            bedrock_client (boto3.client): Boto3 client for interacting with the 'bedrock-agent-runtime' service.
            bedrock_agent_client (boto3.client): Boto3 client for interacting with the 'bedrock-agent' service.

        Raises:
            EnvironmentError: If the 'AWS_REGION' environment variable is not set.
        """

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
