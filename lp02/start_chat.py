import logging

from langchain_core.messages import HumanMessage
import streamlit as st

from cw_expert import CW_GRAPH_RUNNER, CW_SYSTEM_MESSAGE, CwState
from cw_expert.graph import cw_state_to_json
from utilities.logging import configure_logging
from utilities.ux import stringify_simplified_history

configure_logging("./debug.log", "./info.log")

logger = logging.getLogger(__name__)


# Set page configuration to 'wide' to use the full width of the screen
st.set_page_config(layout="wide")

# Initialize session state to keep track of conversation history and LLM messages
if 'conversation' not in st.session_state:
    st.session_state.conversation = []
if 'graph_state' not in st.session_state:
    st.session_state.graph_state = CwState(
        cw_turns = [CW_SYSTEM_MESSAGE],
        approval_turns = [],
        approval_in_progress=False
    )

st.title("Validation Librarian")

# Apply custom CSS for column widths (optional, if further adjustments are needed)
st.markdown(
    """
    <style>
    .left-col {
        width: 600px; /* Fixed width for the left column */
    }
    .right-col {
        width: calc(100% - 600px); /* Remaining space for the right column */
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Create two columns using the default Streamlit column layout
left_col, right_col = st.columns([1, 2])  # Placeholder ratios for the columns

# User input area on the left side
with left_col:
    with st.form(key='user_input_form', clear_on_submit=True):
        user_input = st.text_area("You:", placeholder="Type your message here...", height=100)
        submit_button = st.form_submit_button(label='Send')

# When the user submits a message
if submit_button and user_input:
    # Invoke the LLM with the user input
    next_human_message = HumanMessage(content=user_input)

    approval_in_progress = st.session_state.graph_state.get("approval_in_progress", False)
    logging.info(f"Approval in progress: {approval_in_progress}")

    if approval_in_progress:
        logging.info("Adding human message to approval turns")
        st.session_state.graph_state["approval_turns"].append(next_human_message)
    else:
        logging.info("Adding human message to cw turns")
        st.session_state.graph_state["cw_turns"].append(next_human_message)

    final_state = CW_GRAPH_RUNNER(
        st.session_state.graph_state,
        42
    )

    approval_in_progress = final_state.get("approval_in_progress", False)
    logging.info(f"Approval in progress: {approval_in_progress}")
    is_handoff = final_state.get("is_approval_handoff", False)
    logging.info(f"Is handoff: {is_handoff}")

    if is_handoff:
        logging.info("Pulling AI response from cw turns")
        turns = final_state.get("cw_turns", [])
    elif approval_in_progress:
        logging.info("Pulling AI response from approval turns")
        turns = final_state.get("approval_turns", [])
    else:
        logging.info("Pulling AI response from cw turns")
        turns = final_state.get("cw_turns")

    ai_response = turns[-1]

    # Update the graph state
    logger.debug(f"End of session graph state: {cw_state_to_json(final_state)}")
    st.session_state.graph_state = final_state

    # Add the User Input and LLM response to the chat history, but ensure they are at the top for easy reading
    st.session_state.conversation.insert(0, "---")
    st.session_state.conversation.insert(0, ai_response.content)
    st.session_state.conversation.insert(0, "**-- AI --**")
    st.session_state.conversation.insert(0, user_input)
    st.session_state.conversation.insert(0, "**-- You --**")

# Conversation log on the right side
with right_col:
    # Display the conversation history
    for entry in st.session_state.conversation:
        st.markdown(entry)

