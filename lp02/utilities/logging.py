from functools import wraps
import logging
from typing import Callable, Dict, Any

logger = logging.getLogger(__name__)

def configure_logging(output_file: str, log_level: int = logging.DEBUG):
    logging.basicConfig(
        filename=output_file,
        level=log_level,
        format='%(message)s',  # No prefix, just the message
        filemode='w'  # Overwrite mode
    )
    logging.getLogger("boto3").setLevel(logging.WARNING)
    logging.getLogger("botocore").setLevel(logging.WARNING)


def trace_node(func: Callable) -> Callable:
    """
    Decorator to log method entry at info level and input/output at debug level.
    """
    @wraps(func)
    def wrapper(*args, **kwargs) -> Dict[str, Any]:
        logging.info(f"Entering node: {func.__name__}")
        logging.debug(f"Input args: {args}, kwargs: {kwargs}")
        
        result = func(*args, **kwargs)
        
        logging.debug(f"Output of {func.__name__}: {result}")
        
        return result
    
    return wrapper