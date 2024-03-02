from chatbot import QAChain
import streamlit as st
import os
# from chatbot_w_memory import ConversationalRetrievalChain
from ultimate_chatbot import ConversationalRetrievalChain

import subprocess
import logging

def main():
    # App title
    st.set_page_config(page_title="🦜🔗 Chat with your Data 💬")

    app_tabs = dict()
    tab1, tab2, tab3 = st.tabs(["Upload PDF File", "Enter YouTube URL", "Enter Wikipedia URL"])

    with tab2:
        url_input = st.text_input("YouTube URL", placeholder="Enter the website URL here...")
        app_tabs['youtube'] = url_input

    with tab3:
        url_input = st.text_input("Wikipedia URL", placeholder="Enter the website URL here...")
        app_tabs['wikipedia'] = url_input

    if 'youtube' in app_tabs and isinstance(app_tabs['youtube'], str) and app_tabs['youtube'].strip():
        url = app_tabs.get('youtube')
        data_format = 'youtube'

    if 'wikipedia' in app_tabs and isinstance(app_tabs['wikipedia'], str) and app_tabs['wikipedia'].strip():
        url = app_tabs.get('wikipedia')
        data_format = 'wikipedia'

    # Replicate Credentials
    with st.sidebar:
        st.title('🦜🔗 Chat with your Data 💬')
        if 'OPENAI_API_KEY' in st.secrets:
            st.success('API key already provided!', icon='✅')
            replicate_api = st.secrets['OPENAI_API_KEY']
        else:
            replicate_api = st.text_input('Enter Replicate API token:', type='password')
            if not (replicate_api.startswith('sk-') and len(replicate_api)==51):
                st.warning('Invalid credentials! Please try again', icon='⚠️')
            else:
                st.success('Proceed to entering your prompt message!', icon='👉')
        st.markdown('📖 Learn how to build this app in this [blog](https://blog.streamlit.io/how-to-build-a-llama-2-chatbot/)!')
        st.markdown('📖 Follow me for more [Github](https://github.com/sulaiman-shamasna)!')
    os.environ['OPENAI_API_KEY'] = replicate_api

    # Store LLM generated responses
    if "messages" not in st.session_state.keys():
        st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]

    # Display or clear chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    def clear_chat_history():
        st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]
    st.sidebar.button('Clear Chat History', on_click=clear_chat_history)


    def generate_openai_response(prompt_input):
        qa_chain = ConversationalRetrievalChain().create_chain(data_format, url)
        result = qa_chain({"query": prompt_input})
        return result['result']

    # User-provided prompt
    if prompt := st.chat_input(disabled=not replicate_api):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

    # Generate a new response if last message is not from assistant
    if st.session_state.messages[-1]["role"] != "assistant":
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = generate_openai_response(prompt)
                placeholder = st.empty()
                placeholder.markdown(response)
        message = {"role": "assistant", "content": response}
        st.session_state.messages.append(message)

if __name__ == "__main__":
    main()