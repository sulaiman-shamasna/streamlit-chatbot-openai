# Chat with your Data
---
This chatbot is implemented to enable the user to perform a conversation with their own data in different format, i.e., pdf files, web articles, and youtube videos.

This app uses LangChain to perform the prompting and conversation in the backend. In addition, as a frontend platform streamlit is used.

## Setting up working environment

There are to main ways to use this app:

- **With Docker Container**:

    To build and rnu the docker container, please navigate to the project directory where the ```Dockerfile```  exists, and do the following in the terminal:
    ```py
    docker buuild -t MY_IMAGE .                 # to build the docker container
    docker container run -p 8501:8501 MY_IMAGE  # to run the docker container
    ```
    Note that the port is adjustable, but make sure to choose an unlocated one.

    Having done this, the app will be running on the specified port, open a browser and type the following:
    ```py
    http://localhost:8501/          # or
    http://127.0.0.1:8501/          # Note the port is adjustable
    ```

- **Locally with Virtual Environment**

    Navigate to the project diretory, create a virtual environment using the command:
    ```py
    python -m venv env
    ```
    And activate it, using the command:
    ```py
    - source env/Scripts/activate   # for Windows in the git bash
    - source env/bin/activate       # for Linux and OSX
    ```

    Having done this, run the streamli app using the following command:
    ```py
    streamlit run streamlit_app_v2.py --server.port=8501
    ```

## Streamlit Usage
Having done the previous steps, its time to chat. Follow the following steps, please.

1. Enter a valid OPENAI_API_KEY and press enter
2. Select the tab corresponds to the data type you want to use, i.e., pdf file, web artical, or youtube video. Note you're allowed to use only one data type at once (for the moment).
3. In so being done, you can start chatting.

## Tasks

