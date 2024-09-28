from typing import Annotated, Dict, List, Literal
from typing_extensions import TypedDict

from langchain_aws import ChatBedrockConverse
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.graph.state import CompiledGraph

from approval_expert.tools import TOOLS_ALL, TOOLS_TERMINAL


# Define our LLM
llm = ChatBedrockConverse(
    model="anthropic.claude-3-5-sonnet-20240620-v1:0",
    temperature=0,
    max_tokens=4096,
    region_name="us-west-2"
)
llm_with_tools = llm.bind_tools(TOOLS_ALL)

# Define the state our graph will be operating on
class ApprovalState(TypedDict):
    turns: Annotated[List[BaseMessage], add_messages]
    final_outcome: str

# Set up our tools
tools_terminal_by_name = {tool.name: tool for tool in TOOLS_TERMINAL}

# Define our Graph
approval_graph = StateGraph(ApprovalState)

# Set up our graph nodes
def invoke_llm(state: ApprovalState):
    """
    Node to call the LLM with the current context
    """
    turns = state['turns']
    response = llm_with_tools.invoke(turns)
    # We return a list, because this will get added to the existing list
    return {"turns": [response]}

def terminal_decision(state: ApprovalState):
    """
    Node to invoke the terminal tool and store the decision made by the LLM in the state
    """
    result = []
    tool_call = state["turns"][-1].tool_calls[-1]
    tool = tools_terminal_by_name[tool_call["name"]]
    decision = tool.invoke(tool_call["args"])
    result.append(ToolMessage(content=decision, tool_call_id=tool_call["id"], name=tool.name))
    result.append(AIMessage(content=decision))
    return {"turns": result, "final_outcome": decision}

approval_graph.add_node("invoke_llm", invoke_llm)
approval_graph.add_node("terminal", terminal_decision)

# Define our graph edges
def next_node(state: ApprovalState) -> Literal["terminal", END]:
    """
    Function to route to the correct next node based on the outcome of the previous one
    """
    turns = state['turns']
    last_turn = turns[-1]
    # If the LLM reached a final determination on approval, we route to the "terminal" node
    if last_turn.tool_calls and last_turn.tool_calls[-1]["name"] in tools_terminal_by_name.keys():
        return "terminal"
    # Otherwise, we stop (reply to the user)
    return END

approval_graph.add_edge(START, "invoke_llm")

approval_graph.add_conditional_edges(
    "invoke_llm",
    next_node
)
approval_graph.add_edge("terminal", END)

# Initialize memory to persist state between graph runs
checkpointer = MemorySaver()

# Finally, compile the graph into a LangChain Runnable
APPROVAL_GRAPH = approval_graph.compile(checkpointer=checkpointer)

def _create_runner(workflow: CompiledGraph):
    def run_workflow(turns: List[BaseMessage], thread: int) -> Dict[str, any]:
        events = workflow.stream(
            {"turns": turns},
            config={"configurable": {"thread_id": thread}},
            stream_mode="values"
        )

        final_event = None
        for event in events:
            if "turns" in event:
                event["turns"][-1].pretty_print()
            final_event = event

        return final_event

    return run_workflow

APPROVAL_GRAPH_RUNNER = _create_runner(APPROVAL_GRAPH)