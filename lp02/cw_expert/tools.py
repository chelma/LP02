from dataclasses import dataclass
import logging
import re
import uuid

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from aws_interactions.aws_client_provider import AwsClientProvider

logger = logging.getLogger(__name__)

#
# Define tools to list the metrics for an Amazon OpenSearch Service domain
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


def get_raw_metric_names_for_opensearch_domain(domain_arn: str) -> str:
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
    final_response = f"""
    Metrics for OpenSearch domain '{domain_details.domain_arn}':

    {", ".join(metric_names)}
    """
    return final_response

class PrintRawMetricNamesForOpenSearchDomainArgs(BaseModel):
    """Returns a listing of the raw metric names for an Amazon OpenSearch Service domain directly to the User.  DO NOT USE UNLESS SPECIFICALLY REQUESTED."""
    domain_arn: str = Field(description="The full Amazon ARN of the domain.")

list_raw_metrics_for_opensearch_domain_tool = StructuredTool.from_function(
    func=get_raw_metric_names_for_opensearch_domain,
    name="PrintRawMetricNamesForOpenSearchDomain",
    args_schema=PrintRawMetricNamesForOpenSearchDomainArgs
)

class ExplainMetricsForOpenSearchDomainArgs(BaseModel):
    """PREFERRED way to List, Explain, or Explore the metrics available for an Amazon OpenSearch Service domain.  Queries the CloudWatch API to get a list of metrics then passes it to the LLM to consider."""
    domain_arn: str = Field(description="The full Amazon ARN of the domain.")

explain_metrics_for_opensearch_domain_tool = StructuredTool.from_function(
    func=get_raw_metric_names_for_opensearch_domain,
    name="ExplainMetricsForOpenSearchDomain",
    args_schema=ExplainMetricsForOpenSearchDomainArgs
)

def create_new_cloudwatch_dashboard_from_json(dashboard_json: str, aws_region_name: str) -> str:
    """
    Create a new CloudWatch dashboard from a JSON string.  Returns the ARN of the created dashboard.
    """
    aws_client_provider = AwsClientProvider(aws_region=aws_region_name)
    cloudwatch_client = aws_client_provider.get_cloudwatch()

    # Create the dashboard with a random name
    random_name = f"dashboard-{uuid.uuid4().hex}"
    response = cloudwatch_client.put_dashboard(DashboardName=random_name, DashboardBody=dashboard_json)
    print(response)

    # Return the ARN of the created dashboard
    dashboard_arn = cloudwatch_client.get_dashboard(DashboardName=random_name)["DashboardArn"]
    
    return dashboard_arn

class CreateNewCloudwatchDashboardFromJsonArgs(BaseModel):
    """Creates a completely new CloudWatch dashboard from a JSON string, assigning it a random UUID for a name. Returns the ARN of the created dashboard.  Do not invoke without first having shown the Dashboard JSON to the human operator."""
    dashboard_json: str = Field(description="The JSON string representing the new dashboard's full body definition.")
    aws_region_name: str = Field(description="The AWS Region to create the dashboard in (example: us-east-1, us-west-2, etc...).")

create_new_cloudwatch_dashboard_from_json_tool = StructuredTool.from_function(
    func=create_new_cloudwatch_dashboard_from_json,
    name="CreateNewCloudwatchDashboardFromJson",
    args_schema=CreateNewCloudwatchDashboardFromJsonArgs
)

TOOLS_NORMAL = [explain_metrics_for_opensearch_domain_tool]
TOOLS_DIRECT_RESPONSE = [list_raw_metrics_for_opensearch_domain_tool]
TOOLS_NEED_APPROVAL = [create_new_cloudwatch_dashboard_from_json_tool]
TOOLS_ALL = TOOLS_NORMAL + TOOLS_DIRECT_RESPONSE + TOOLS_NEED_APPROVAL
