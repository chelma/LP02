
from langchain_core.messages import HumanMessage

from cw_expert import CW_AGENT, CW_SYSTEM_MESSAGE

llm_messages = [
    CW_SYSTEM_MESSAGE,
    HumanMessage(content="What metrics are there for OpenSearch domain arn:aws:es:us-west-2:729929230507:domain/arkimedomain872-vzfrvtegjekp ?")
]


final_state = CW_AGENT.invoke(
    {"messages": llm_messages},
    config={"configurable": {"thread_id": 42}}
)


print(final_state)