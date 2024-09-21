import logging
from typing import Dict

import boto3

logger = logging.getLogger(__name__)

class AwsClientProvider:
    def __init__(self, aws_profile: str = "default", aws_region: str = None, aws_compute=False):
        """
        Wrapper around creation of Boto AWS Clients.
        aws_profile: if not provided, will use "default"
        aws_region: if not provided, will use the default region in your local AWS Config
        aws_compute: if True, will not use the profile, and will use the local AWS Config
        """
        self._aws_profile = aws_profile
        self._aws_region = aws_region
        self._aws_compute = aws_compute
    
    def _get_assumed_credentials(self, current_session: boto3.Session) -> Dict[str, str]:
        sts_client = current_session.client("sts")

        # Assume the role in the target account
        assumed_role_object = sts_client.assume_role(
            RoleArn=self._assume_role_arn,
            RoleSessionName="ArkimeAwsAioCLI"
        )

        return assumed_role_object["Credentials"]

    def _get_session(self) -> boto3.Session:
        if self._aws_compute:
            current_account_session = boto3.Session()
        else:
            current_account_session = boto3.Session(profile_name=self._aws_profile, region_name=self._aws_region)
        
        return current_account_session

    def get_acm(self):
        session = self._get_session()
        client = session.client("acm")
        return client

    def get_cloudwatch(self):
        session = self._get_session()
        client = session.client("cloudwatch")
        return client 

    def get_ec2(self):
        session = self._get_session()
        client = session.client("ec2")
        return client 

    def get_ecs(self):
        session = self._get_session()
        client = session.client("ecs")
        return client

    def get_events(self):
        session = self._get_session()
        client = session.client("events")
        return client

    def get_iam(self):
        session = self._get_session()
        client = session.client("iam")
        return client

    def get_opensearch(self):
        session = self._get_session()
        client = session.client("opensearch")
        return client

    def get_s3(self):
        session = self._get_session()
        client = session.client("s3")
        return client

    def get_s3_resource(self):
        boto3.setup_default_session(profile_name=self._aws_profile)
        resource = boto3.resource("s3", region_name=self._aws_region)
        return resource

    def get_secretsmanager(self):
        session = self._get_session()
        client = session.client("secretsmanager")
        return client

    def get_ssm(self):
        session = self._get_session()
        client = session.client("ssm")
        return client

    def get_sts(self):
        session = self._get_session()
        client = session.client("sts")
        return client