
from langchain_core.messages import HumanMessage

from cw_expert import CW_AGENT, CW_SYSTEM_MESSAGE
from utilities.ux import stringify_simplified_history

llm_messages = [
    CW_SYSTEM_MESSAGE,
    HumanMessage(content="Give me the raw metric names for the OpenSearch domain arn:aws:es:us-west-2:729929230507:domain/arkimedomain872-vzfrvtegjekp")
]

final_state = CW_AGENT.invoke(
    {"messages": llm_messages},
    config={"configurable": {"thread_id": 42}}
)
print(stringify_simplified_history(final_state["messages"]))



# llm_messages = final_state["messages"]
# llm_messages.append(HumanMessage(content="Great, but can you just list me the raw metric names?"))
# final_state = CW_AGENT.invoke(
#     {"messages": llm_messages},
#     config={"configurable": {"thread_id": 42}}
# )
# print(stringify_simplified_history(final_state["messages"]))

llm_messages = final_state["messages"]
llm_messages.append(HumanMessage(content="Which of these metrics would be useful for determining whether the Domain is handling a large write volume gracefully?"))
final_state = CW_AGENT.invoke(
    {"messages": llm_messages},
    config={"configurable": {"thread_id": 42}}
)
print(stringify_simplified_history(final_state["messages"]))