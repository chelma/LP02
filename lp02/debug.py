import logging

from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

from approval_expert import get_approval_expert_system_message, APPROVAL_GRAPH_RUNNER
from cw_expert import CW_GRAPH, CW_GRAPH_RUNNER, CW_SYSTEM_MESSAGE, CwState
from utilities.ux import stringify_simplified_history
from utilities.logging import configure_logging

configure_logging("./debug.log", "./info.log")

logger = logging.getLogger(__name__)

# logger.info(HumanMessage(content="Hello!").to_json())
# logger.info(AIMessage(content="Hello!").to_json())
# logger.info(ToolMessage(name="TestTool", content="Hello!", tool_call_id="id").to_json())

# turns = [
#     get_approval_expert_system_message("Pour out the old coffee"),
#     HumanMessage(content="Why is the coffee old?")
# ]

# final_state = APPROVAL_GRAPH_RUNNER(turns, 42)
# print(stringify_simplified_history(final_state["turns"]))

# final_state["turns"].append(HumanMessage(content="Nah"))  
# final_state = APPROVAL_GRAPH_RUNNER(final_state["turns"], 42)
# print(stringify_simplified_history(final_state["turns"]))

# cw_state = CwState(
#     cw_turns = [
#         CW_SYSTEM_MESSAGE,
#         HumanMessage(content="What can you do?")
#     ],
#     approval_in_progress=False
# )
# final_state = CW_GRAPH_RUNNER(cw_state, 42)
# # print(stringify_simplified_history(final_state["cw_turns"]))

# final_state["cw_turns"].append(
#     HumanMessage(content="What metrics do you have for the OpenSearch domain arn:aws:es:us-west-2:729929230507:domain/arkimedomain872-vzfrvtegjekp?")
# )
# final_state = CW_GRAPH_RUNNER(final_state, 42)
# # print(stringify_simplified_history(final_state["cw_turns"]))


# final_state["cw_turns"].append(
#     HumanMessage(content="Can you just give me a list of the raw metric names?")
# )
# final_state = CW_GRAPH_RUNNER(final_state, 42)
# # print(stringify_simplified_history(final_state["cw_turns"]))

# final_state["cw_turns"].append(
#     HumanMessage(content=("Can you please give me the JSON for a CloudWatch Dashboard that shows two graphs:"
#                           + "\n\n1. Average IndexingRate, WriteIOPS, and IndexingLatency"
#                           + "\n2. P99 IndexingRate, CPUUtilization, and JVMMemoryPressure"
#                           + "\n\nDo not create the Dashboard, just give me the JSON"))
# )
# final_state = CW_GRAPH_RUNNER(final_state, 42)
# # print(stringify_simplified_history(final_state["cw_turns"]))

# final_state["cw_turns"].append(
#     HumanMessage(content="Cool!  Now make that Dashboard, please.")
# )
# final_state = CW_GRAPH_RUNNER(final_state, 42)
# # print(stringify_simplified_history(final_state["cw_turns"]))

# final_state["cw_turns"].append(
#     HumanMessage(content="Approved!")
# )
# final_state = CW_GRAPH_RUNNER(final_state, 42)


# cw_state = CwState(
#     cw_turns = [
#         CW_SYSTEM_MESSAGE
#     ],
#     approval_in_progress=False
# )
# cw_state["cw_turns"].append(
#     HumanMessage(content="Can you please make me a CloudWatch Dashboard for the Domain domain arn:aws:es:us-west-2:729929230507:domain/arkimedomain872-vzfrvtegjekp that shows me a single graph with the Average IndexingRate, WriteIOPS, and IndexingLatency?")
# )
# final_state = CW_GRAPH_RUNNER(cw_state, 42)

# final_state["approval_turns"].append(
#     HumanMessage(content="Yeah?")
# )
# final_state = CW_GRAPH_RUNNER(final_state, 42)

# final_state["approval_turns"].append(
#     HumanMessage(content="OK, sure, create it")
# )
# final_state = CW_GRAPH_RUNNER(final_state, 42)

# Setting xray to 1 will show the internal structure of the nested graph
CW_GRAPH.get_graph(xray=1).draw_mermaid_png(output_file_path="cw_graph.png")