
from langchain_core.messages import HumanMessage

from approval_expert import get_approval_expert_system_message, APPROVAL_GRAPH_RUNNER
from cw_expert import CW_AGENT, CW_SYSTEM_MESSAGE
from utilities.ux import stringify_simplified_history


turns = [
    get_approval_expert_system_message("Pour out the old coffee"),
    HumanMessage(content="Why is the coffee old?")
]

final_state = APPROVAL_GRAPH_RUNNER(turns, 42)
print(stringify_simplified_history(final_state["turns"]))

# final_state["turns"].append(HumanMessage(content="Nah"))  
# final_state = APPROVAL_GRAPH_RUNNER(final_state["turns"], 42)
# print(stringify_simplified_history(final_state["turns"]))

# llm_messages = [
#     CW_SYSTEM_MESSAGE,
#     HumanMessage(content="Give me the raw metric names for the OpenSearch domain arn:aws:es:us-west-2:729929230507:domain/arkimedomain872-vzfrvtegjekp")
# ]

# final_state = CW_AGENT.invoke(
#     {"messages": llm_messages},
#     config={"configurable": {"thread_id": 42}}
# )
# print(stringify_simplified_history(final_state["messages"]))



# llm_messages = final_state["messages"]
# llm_messages.append(HumanMessage(content="Great, but can you just list me the raw metric names?"))
# final_state = CW_AGENT.invoke(
#     {"messages": llm_messages},
#     config={"configurable": {"thread_id": 42}}
# )
# print(stringify_simplified_history(final_state["messages"]))

# llm_messages = final_state["messages"]
# llm_messages.append(HumanMessage(content="Which of these metrics would be useful for determining whether the Domain is handling a large write volume gracefully?"))
# final_state = CW_AGENT.invoke(
#     {"messages": llm_messages},
#     config={"configurable": {"thread_id": 42}}
# )
# print(stringify_simplified_history(final_state["messages"]))