
from langchain_core.messages import HumanMessage
import streamlit as st

from cw_expert import CW_GRAPH_RUNNER, CW_SYSTEM_MESSAGE
from utilities.ux import stringify_simplified_history


# Set page configuration to 'wide' to use the full width of the screen
st.set_page_config(layout="wide")

# Initialize session state to keep track of conversation history and LLM messages
if 'conversation' not in st.session_state:
    st.session_state.conversation = []
if 'llm_messages' not in st.session_state:
    st.session_state.llm_messages = [CW_SYSTEM_MESSAGE]

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
    st.session_state.llm_messages.append(HumanMessage(content=user_input))
    final_state = CW_GRAPH_RUNNER(
        st.session_state.llm_messages,
        42
    )
    ai_response = final_state["turns"][-1]

    print("=======================================================================================================")
    print(stringify_simplified_history(final_state["turns"]))

    # Update the message history
    st.session_state.llm_messages = final_state["turns"]

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

# print(st.session_state.llm_messages)
