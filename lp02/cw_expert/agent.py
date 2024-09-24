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

# Set up our tools
llm_with_tools = llm.bind_tools(TOOLS_ALL)
tool_node = ToolNode(TOOLS_NORMAL)
tools_direct_by_name = {tool.name: tool for tool in TOOLS_DIRECT_RESPONSE}

# Handle tool calls which require a "direct" final response to the user a bit differently
# We MUST have an AIMessage between each ToolMessage and any HumanMessage.  We spoof that
# by adding an AIMessage at the end of our calls here.
def tool_node_direct(state: dict):
    result = []
    tool_call = state["messages"][-1].tool_calls[-1]
    tool = tools_direct_by_name[tool_call["name"]]
    observation = tool.invoke(tool_call["args"])
    result.append(ToolMessage(content=observation, tool_call_id=tool_call["id"]))
    result.append(AIMessage(content=observation))

    return {"messages": result}

# Define the function that determines whether to continue or not
def should_continue(state: MessagesState) -> Literal["tool_direct", "tool", END]:
    messages = state['messages']
    last_message = messages[-1]
    # If the LLM makes a tool call, then we route to the "tools" node unless we need a direct response
    if last_message.tool_calls and last_message.tool_calls[-1]["name"] in tools_direct_by_name.keys():
        return "tool_direct"
    elif last_message.tool_calls:
        return "tool"
    # Otherwise, we stop (reply to the user)
    return END

# Define the function that calls the model
def call_model(state: MessagesState):
    messages = state['messages']
    response = llm_with_tools.invoke(messages)
    # We return a list, because this will get added to the existing list
    return {"messages": [response]}

# Define a new graph
workflow = StateGraph(MessagesState)

# Define the nodes; we cycle between the agent and the tool nodes unless we need a direct response from a tool
workflow.add_node("agent", call_model)
workflow.add_node("tool", tool_node)
workflow.add_node("tool_direct", tool_node_direct)

# Set the entrypoint as `agent`
# This means that this node is the first one called
workflow.add_edge(START, "agent")

# We now add a conditional edge
workflow.add_conditional_edges(
    # First, we define the start node. We use `agent`.
    # This means these are the edges taken after the `agent` node is called.
    "agent",
    # Next, we pass in the function that will determine which node is called next.
    should_continue,
)

# We now add a normal edge from `tools` to `agent`.
# This means that after `tools` is called, `agent` node is called next.
workflow.add_edge("tool", 'agent')

# We now add a normal edge to ensure that the direct response is sent to the user
workflow.add_edge("tool_direct", END)

# Initialize memory to persist state between graph runs
checkpointer = MemorySaver()

# Finally, we compile it!
# This compiles it into a LangChain Runnable,
# meaning you can use it as you would any other runnable.
# Note that we're (optionally) passing the memory when compiling the graph
AGENT = workflow.compile(checkpointer=checkpointer)

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

AGENT_RUNNER = _create_runner(AGENT)