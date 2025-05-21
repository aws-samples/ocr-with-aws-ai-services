import boto3
import logging
from functools import lru_cache

# Configure logging
logger = logging.getLogger(__name__)

@lru_cache(maxsize=16)
def get_aws_client(service_name, region=None, endpoint_url=None):
    """
    Get a cached boto3 client to avoid creating new connections
    
    Args:
        service_name: AWS service name
        region: AWS region name
        endpoint_url: Optional custom endpoint URL
        
    Returns:
        Boto3 client for the requested service
    """
    logger.debug(f"Getting AWS client for {service_name} in region {region}")
    kwargs = {}
    if region:
        kwargs['region_name'] = region
    if endpoint_url:
        kwargs['endpoint_url'] = endpoint_url
        
    return boto3.client(service_name, **kwargs)

@lru_cache(maxsize=8)
def get_aws_session(region=None):
    """
    Get a cached boto3 session
    
    Args:
        region: AWS region name
        
    Returns:
        Boto3 session
    """
    if region:
        return boto3.session.Session(region_name=region)
    return boto3.session.Session()

def get_account_id():
    """
    Get the current AWS account ID
    
    Returns:
        AWS account ID
    """
    sts_client = get_aws_client('sts')
    return sts_client.get_caller_identity()['Account']

def get_current_region():
    """
    Get the current AWS region
    
    Returns:
        AWS region name
    """
    return get_aws_session().region_name