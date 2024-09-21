from typing import Literal

from langchain_aws import ChatBedrockConverse
from langchain_core.messages import ToolMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph, MessagesState
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
tool_node = ToolNode(TOOLS_ALL)
tool_names_direct = [tool.name for tool in TOOLS_DIRECT_RESPONSE]

# Define the function that determines whether to continue or not
def should_continue(state: MessagesState) -> Literal["tool_direct", "tool", END]:
    messages = state['messages']
    last_message = messages[-1]
    # If the LLM makes a tool call, then we route to the "tools" node unless we need a direct response
    print(last_message.tool_calls)
    if last_message.tool_calls and last_message.tool_calls[-1]["name"] in tool_names_direct:
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
workflow.add_node("tool_direct", tool_node)

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
