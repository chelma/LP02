from typing import Dict, List, Literal

from langchain_aws import ChatBedrockConverse
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph, MessagesState
from langgraph.graph.state import CompiledGraph
from langgraph.prebuilt import ToolNode

from cw_expert.tools import TOOLS_ALL, TOOLS_DIRECT_RESPONSE, TOOLS_NORMAL


# Define our LLM
llm = ChatBedrockConverse(
    model="anthropic.claude-3-5-sonnet-20240620-v1:0",
    temperature=0,
    max_tokens=4096,
    region_name="us-west-2"
)
llm_with_tools = llm.bind_tools(TOOLS_ALL)

# Define the state our graph will be operating on
# TODO

# Set up our tools
tools_direct_by_name = {tool.name: tool for tool in TOOLS_DIRECT_RESPONSE}

# Define our graph
cw_graph = StateGraph(MessagesState)

# Set up our graph nodes
tool_node = ToolNode(TOOLS_NORMAL)

def tool_node_direct(state: MessagesState):
    """
    Handle tool calls which require a "direct" final response to the user a bit differently.
    We MUST have an AIMessage between each ToolMessage and any HumanMessage.  We spoof that
    by adding an AIMessage at the end of our calls here.
    """
    result = []
    tool_call = state["messages"][-1].tool_calls[-1]
    tool = tools_direct_by_name[tool_call["name"]]
    observation = tool.invoke(tool_call["args"])
    result.append(ToolMessage(content=observation, tool_call_id=tool_call["id"]))
    result.append(AIMessage(content=observation))

    return {"messages": result}

def invoke_llm(state: MessagesState):
    messages = state['messages']
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}

cw_graph.add_node("invoke_llm", invoke_llm)
cw_graph.add_node("tool", tool_node)
cw_graph.add_node("tool_direct", tool_node_direct)

# Define our graph edges
def next_node(state: MessagesState) -> Literal["tool_direct", "tool", END]:
    messages = state['messages']
    last_message = messages[-1]
    # If the LLM makes a tool call, then we route to the "tools" node unless we need a direct response
    if last_message.tool_calls and last_message.tool_calls[-1]["name"] in tools_direct_by_name.keys():
        return "tool_direct"
    elif last_message.tool_calls:
        return "tool"
    return END


cw_graph.add_edge(START, "invoke_llm")

cw_graph.add_conditional_edges(
    "invoke_llm",
    next_node,
)

cw_graph.add_edge("tool", 'invoke_llm')

cw_graph.add_edge("tool_direct", END)

# Initialize memory to persist state between graph runs
checkpointer = MemorySaver()

# Finally, compile the graph into a LangChain Runnable
CW_GRAPH = cw_graph.compile(checkpointer=checkpointer)

def _create_runner(workflow: CompiledGraph):
    def run_workflow(messages: List[BaseMessage], thread: int) -> Dict[str, any]:
        events = workflow.stream(
            {"messages": messages},
            config={"configurable": {"thread_id": thread}},
            stream_mode="values"
        )

        final_event = None
        for event in events:
            if "messages" in event:
                event["messages"][-1].pretty_print()
            final_event = event

        return final_event

    return run_workflow

CW_GRAPH_RUNNER = _create_runner(CW_GRAPH)