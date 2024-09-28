import logging

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

#
# Define dummy tools that the approval expert can "invoke" to indicate the end of an the interaction
#

def approval_granted() -> str:
    return "The human operator approved the operation."

approval_granted_tool = StructuredTool.from_function(
    func=approval_granted,
    name="ApprovalGranted",
    description="Invoke to indicate that the human operator has approved the operation."
)

def approval_denied() -> str:
    return "The human operator denied the operation."

approval_denied_tool = StructuredTool.from_function(
    func=approval_denied,
    name="ApprovalDenied",
    description="Invoke to indicate that the human operator has denied approval for the operation."
)

def approval_other(human_operator_response: str) -> str:
    return f"The human operator changed the topic or needs more information.  Here's what they said:\n\n<human_response>{human_operator_response}</human_response>"

class ApprovalOtherArgs(BaseModel):
    human_operator_response: str = Field(description="The full response from the human operator.")

approval_other_tool = StructuredTool.from_function(
    func=approval_other,
    name="ApprovalOther",
    description="Invoke to indicate that the human operator's response is unclear, off-topic, or needs more information."
)

TOOLS_TERMINAL = [approval_granted_tool, approval_denied_tool, approval_other_tool]
TOOLS_ALL = TOOLS_TERMINAL
