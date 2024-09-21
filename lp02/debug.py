
from aws_interactions.aws_client_provider import AwsClientProvider
from cw_expert.tools import parse_domain_arn


domain_details = parse_domain_arn("arn:aws:es:us-west-2:729929230507:domain/arkimedomain872-vzfrvtegjekp")


aws_client_provider = AwsClientProvider(aws_region=domain_details.region)
cloudwatch_client = aws_client_provider.get_cloudwatch()

# Pull metrics names until we don't have a NextToken

metric_names = []
next_token = None
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

metric_names.sort()
print(metric_names)