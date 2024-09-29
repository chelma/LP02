import logging
from typing import Annotated, Dict, List, Literal
from typing_extensions import TypedDict

from langchain_aws import ChatBedrockConverse
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.graph.state import CompiledGraph
from langgraph.prebuilt import ToolNode

from approval_expert import APPROVAL_GRAPH, get_approval_expert_system_message
from cw_expert.tools import TOOLS_ALL, TOOLS_DIRECT_RESPONSE, TOOLS_NORMAL, TOOLS_NEED_APPROVAL
from utilities.logging import trace_node
from utilities.graph import add_messages_with_reset, ResetMessages

logger = logging.getLogger(__name__)


# Define our LLM
llm = ChatBedrockConverse(
    model="anthropic.claude-3-5-sonnet-20240620-v1:0",
    temperature=0,
    max_tokens=4096,
    region_name="us-west-2"
)
llm_with_tools = llm.bind_tools(TOOLS_ALL)

# Define the state our graph will be operating on
class CwState(TypedDict):
    # Local to the parent graph
    cw_turns: Annotated[List[BaseMessage], add_messages_with_reset]
    ops_to_approve: Dict[str, any]

    # Used to resume a conversation w/ the approval sub-graph
    approval_in_progress: bool
    is_approval_handoff: bool
    approval_turns: Annotated[List[BaseMessage], add_messages_with_reset]
    approval_outcome: str

# Set up our tools
tools_normal_by_name = {tool.name: tool for tool in TOOLS_NORMAL}
tools_direct_by_name = {tool.name: tool for tool in TOOLS_DIRECT_RESPONSE}
tools_approval_by_name = {tool.name: tool for tool in TOOLS_NEED_APPROVAL}

# Initialize memory to persist state between graph runs
checkpointer = MemorySaver()

# Define our graph
cw_graph = StateGraph(CwState)

# Set up our graph nodes
@trace_node
def tool_node(state: CwState) -> Dict[str, any]:
    """
    Node to handle normal tool cals
    """
    result = []
    tool_call = state["cw_turns"][-1].tool_calls[-1]
    tool = tools_normal_by_name[tool_call["name"]]
    observation = tool.invoke(tool_call["args"])
    result.append(ToolMessage(name=tool_call["name"], content=observation, tool_call_id=tool_call["id"]))

    return {"cw_turns": result}

@trace_node
def tool_node_direct(state: CwState):
    """
    Node to handle tool calls where we need to return the raw response from the tool directly
    to the user rather than passing it through the LLM first.

    This requires special handling, as LLM APIs expect an AIMessage between each ToolMessage
    and any HumanMessage.  We spoof that by adding an AIMessage at the end of our calls here.
    """
    result = []
    tool_call = state["cw_turns"][-1].tool_calls[-1]
    tool = tools_direct_by_name[tool_call["name"]]
    observation = tool.invoke(tool_call["args"])
    result.append(ToolMessage(name="DummyToolNodeDirect", content=observation, tool_call_id=tool_call["id"]))
    result.append(AIMessage(content=observation))

    return {"cw_turns": result}

@trace_node
def tool_node_approval(state: CwState):
    """
    Node to handle calling a tool after it has been manually reviewed and approved by the human
    operator.

    This requires special handling because we need to pull the tool details from the graph state
    rather than the message state.
    """
    tool_call = state["ops_to_approve"][-1]
    tool = tools_approval_by_name[tool_call["name"]]
    observation = tool.invoke(tool_call["args"])
    
    result = ToolMessage(name="DummyToolNodeApproval", content=observation, tool_call_id=tool_call["id"])

    return {"cw_turns": [result]}

@trace_node
def prep_approval(state: CwState):
    """
    Node to prepare the state for the approval sub-graph.  It sets up the the final message to the user from
    the main graph and prepares the sub-graph state to receive the next message from the user.
    """
    # Save the original tool call that triggered the need for approval
    ops_to_approve = state["cw_turns"][-1].tool_calls

    # Create the system message that will be used to prompt the Approval LLM.  We need to reset the LLM 
    # context window here, as we're starting a new approval conversation.
    approval_system_message = get_approval_expert_system_message(str(ops_to_approve))
    reset_turns = ResetMessages(messages=[approval_system_message])

    # Set the message that will be displayed to the user for them to respond with an approval decision
    approval_prompt = AIMessage(content=f"Before I can perform that action, I need your approval.  Please approve or deny the following operation:\n\n{ops_to_approve}")

    return {"cw_turns": [approval_prompt], "ops_to_approve": ops_to_approve, "approval_in_progress": True, "is_approval_handoff": True, "approval_outcome": None, "approval_turns": reset_turns}

@trace_node
def invoke_llm_cw(state: CwState):
    cw_turns = state['cw_turns']
    response = llm_with_tools.invoke(cw_turns)
    return {"cw_turns": [response]}

cw_graph.add_node("invoke_llm_cw", invoke_llm_cw)
cw_graph.add_node("tool", tool_node)
cw_graph.add_node("tool_direct", tool_node_direct)
cw_graph.add_node("prep_approval", prep_approval)
cw_graph.add_node("approval", APPROVAL_GRAPH.compile(checkpointer=checkpointer))
cw_graph.add_node("tool_approval", tool_node_approval)

# Define our graph edges
def starting_node(state: CwState) -> Literal["approval", "invoke_llm_cw"]:
    # Handle the handoff process
    if state.get("is_approval_handoff", False):
        state["is_approval_handoff"] = False
        return "approval"

    # If we're in the middle of an approval conversation, we need to route to the approval sub-graph
    if state.get("approval_in_progress", False):
        return "approval"
    
    # Otherwise, we start with the LLM
    return "invoke_llm_cw"

def next_node(state: CwState) -> Literal["prep_approval", "tool_direct", "tool_approval", "tool", END]:
    cw_turns = state['cw_turns']
    last_message = cw_turns[-1]
    # Route to the tools needing a direct response
    if last_message.tool_calls and last_message.tool_calls[-1]["name"] in tools_direct_by_name.keys():
        return "tool_direct"
    # The tool request needs approval; route accordingly
    if last_message.tool_calls and last_message.tool_calls[-1]["name"] in tools_approval_by_name.keys():
        return "prep_approval"
    elif last_message.tool_calls:
        return "tool"
    return END

def next_node_after_approval(state: CwState) -> Literal["tool_approval", "invoke_llm_cw", END]:
    # We have an approval outcome
    if not state.get("approval_in_progress"):
        approval_outcome = state["approval_outcome"]

        # If the human operator approved the operation, we route to the "tool_approval" node
        if approval_outcome == "ApprovalGranted":
            return "tool_approval"
        
        if approval_outcome == "ApprovalDenied":
            state["cw_turns"].append(
                ToolMessage(
                    name="DummyToolApprovalDenied",
                    content="The human operator denied permission to perform the operation."
                )
            )
            return "invoke_llm_cw"
        
        if approval_outcome == "ApprovalOther":
            last_approval_turn = state["approval_turns"][-1]
            state["cw_turns"].append(
                ToolMessage(
                    name="DummyToolApprovalOther",
                    content=last_approval_turn.content
                )
            )
            return "invoke_llm_cw"
    
    # We're still talking to the human operator about approval
    return END

cw_graph.add_conditional_edges(START, starting_node)
cw_graph.add_conditional_edges("invoke_llm_cw", next_node)
cw_graph.add_conditional_edges("approval", next_node_after_approval)

cw_graph.add_edge("tool", 'invoke_llm_cw')
cw_graph.add_edge("tool_approval", 'invoke_llm_cw')
cw_graph.add_edge("tool_direct", END)
cw_graph.add_edge("prep_approval", END)

# Finally, compile the graph into a LangChain Runnable
CW_GRAPH = cw_graph.compile(checkpointer=checkpointer)

def _create_runner(workflow: CompiledGraph):
    def run_workflow(cw_state: CwState, thread: int) -> CwState:
        states = workflow.stream(
            cw_state,
            config={"configurable": {"thread_id": thread}},
            stream_mode="values"
        )

        final_state = None
        for state in states:
            if "cw_turns" in state:
                state["cw_turns"][-1].pretty_print()
                logger.info(state["cw_turns"][-1].to_json())
            final_state = state

        return final_state

    return run_workflow

CW_GRAPH_RUNNER = _create_runner(CW_GRAPH)