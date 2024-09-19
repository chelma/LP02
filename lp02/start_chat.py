from langchain_aws import ChatBedrock
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
import streamlit as st

# Set page configuration to 'wide' to use the full width of the screen
st.set_page_config(layout="wide")

# Initialize the LLM
llm = ChatBedrock(
    model_id="anthropic.claude-3-5-sonnet-20240620-v1:0",
    model_kwargs={"max_tokens": 20000, "temperature": 0.3},
    region_name="us-west-2"
)

# Initialize session state to keep track of conversation history and LLM messages
if 'conversation' not in st.session_state:
    st.session_state.conversation = []
if 'llm_messages' not in st.session_state:
    st.session_state.llm_messages = [SystemMessage(content="You are a helpful chatbot.")]

st.title("GenAI Chatbot")

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
    # Append the user input to the chat history
    st.session_state.conversation.append(f"**You:** {user_input}")

    # Add the user input to the LLM context
    st.session_state.llm_messages.append(HumanMessage(content=user_input))

    # Invoke the LLM with the user input
    ai_response = llm.invoke(st.session_state.llm_messages)

    # Add the LLM response to the LLM context
    st.session_state.llm_messages.append(AIMessage(content=ai_response.content))

    # Append the LLM response to the chat history
    st.session_state.conversation.append(f"**AI:** {ai_response.content}")

# Conversation log on the right side
with right_col:
    # Display the conversation history
    for entry in st.session_state.conversation:
        st.markdown(entry)

print(st.session_state.llm_messages)
