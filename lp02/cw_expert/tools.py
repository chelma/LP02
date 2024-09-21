from dataclasses import dataclass
import logging
import re

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from aws_interactions.aws_client_provider import AwsClientProvider

logger = logging.getLogger(__name__)

#
# Define a tool to list the metrics for an Amazon OpenSearch Service domain
#

class InvalidDomainArnError(Exception):
    pass

@dataclass
class DomainDetails:
    domain_name: str
    domain_arn: str
    region: str
    account_id: str

def parse_domain_arn(domain_arn: str) -> DomainDetails:
    """
    Parse the domain ARN to extract the relevant details.  Raises if the ARN is not structured correctly.

    Example domain ARN: arn:aws:es:us-west-2:123456789012:domain/my-domain
    """
    # Validate the ARN string format w/ a regex
    arn_regex = re.compile(r"arn:aws:es:(.*?):(.*?):domain/(.*)")
    match = arn_regex.match(domain_arn)
    if not match:
        raise InvalidDomainArnError(f"The ARN '{domain_arn}' does not match the expected format: arn:aws:es:<region>:<account_id>:domain/<domain_name>")

    region, account_id, domain_name = match.groups()
    return DomainDetails(domain_name=domain_name, domain_arn=domain_arn, region=region, account_id=account_id)


def list_metrics_for_opensearch_domain(domain_arn: str) -> str:
    try:
        domain_details = parse_domain_arn(domain_arn)
    except InvalidDomainArnError as e:
        return f"Error: {str(e)}"
    
    print(f"Listing metrics for OpenSearch domain: {domain_details.domain_name}")
    
    aws_client_provider = AwsClientProvider(aws_region=domain_details.region)
    cloudwatch_client = aws_client_provider.get_cloudwatch()
    
    # Pull metrics names until we don't have a NextToken
    metric_names = []
    next_token = None
    try:
        while True:
            args = {
                "Namespace": "AWS/ES",
                "Dimensions": [
                    {"Name": "ClientId", "Value": domain_details.account_id},
                    {"Name": "DomainName", "Value": domain_details.domain_name},
                ]
            }
            if next_token:
                args["NextToken"] = next_token
            response = cloudwatch_client.list_metrics(**args)
            new_metric_names = [metric["MetricName"] for metric in response["Metrics"]]
            print(f"Found {len(new_metric_names)} metrics")

            metric_names.extend(new_metric_names)
            next_token = response.get("NextToken", None)
            if not next_token:
                break

    except Exception as e:
        return f"Error: {str(e)}"
    
    metric_names.sort()
    return ", ".join(metric_names)

class ListMetricsForOpenSearchDomainArgs(BaseModel):
    """Lists the metric names for an Amazon OpenSearch Service domain as a comma-separated string."""
    domain_arn: str = Field(description="The full Amazon ARN of the domain.")

list_metrics_for_opensearch_domain_tool = StructuredTool.from_function(
    func=list_metrics_for_opensearch_domain,
    name="ListMetricsForOpenSearchDomain",
    args_schema=ListMetricsForOpenSearchDomainArgs
)

TOOLS = [list_metrics_for_opensearch_domain_tool]
