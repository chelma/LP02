from dataclasses import dataclass

from typing import Union
from langgraph.graph.message import add_messages, Messages

@dataclass
class ResetMessages:
    messages: Messages

def add_messages_with_reset(left: Messages, right: Union[Messages | ResetMessages]) -> Messages:
    """
    Performs the usual merge behavior, but if the right side is a ResetMessages object, it returns
    its messages directly instead of merging them with the existing messages.
    """
    if isinstance(right, ResetMessages):
        return right.messages
    return add_messages(left, right)