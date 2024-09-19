from langchain_aws import ChatBedrock
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
import streamlit as st

llm = ChatBedrock(
    model_id="anthropic.claude-3-5-sonnet-20240620-v1:0",
    region_name="us-west-2"
)


llm_messages = [
    SystemMessage(content="You are a helpful chatbot.")
]

# Initialize session state to keep track of the conversation history
if 'conversation' not in st.session_state:
    st.session_state.conversation = []

st.title("GenAI Chatbot")

# User input area
user_input = st.text_input("You:", placeholder="Type your message here...")

# When the user submits a message
if user_input:
    # Append the user input to the chat history
    st.session_state.conversation.append(f"You: {user_input}")

    # Add the user input to the LLM context
    llm_messages.append(HumanMessage(content=user_input))

    # Invoke the LLM with the user input
    ai_response = llm.invoke(llm_messages)

    # Add the LLM response to the LLM context
    llm_messages.append(AIMessage(content=ai_response.content))

    # Append the LLM response to the chat history
    st.session_state.conversation.append(f"AI: {ai_response.content}")

    # Clear the input box by setting `st.session_state` value for 'user_input'
    st.session_state.user_input = ""

# Display the conversation history
for entry in st.session_state.conversation:
    st.write(entry)
