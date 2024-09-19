import streamlit as st

# Initialize session state to keep track of the conversation history
if 'conversation' not in st.session_state:
    st.session_state.conversation = []

st.title("GenAI Chatbot")

# User input area
user_input = st.text_input("You:", placeholder="Type your message here...")

# When the user submits a message
if user_input:
    # Append the user input and mock AI response to the conversation history
    st.session_state.conversation.append(f"You: {user_input}")
    st.session_state.conversation.append("AI: <AI response>")

    # Clear the input box by setting `st.session_state` value for 'user_input'
    st.session_state.user_input = ""

# Display the conversation history
for entry in st.session_state.conversation:
    st.write(entry)
