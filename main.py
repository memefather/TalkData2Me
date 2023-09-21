import streamlit as st
import pandas as pd
import openai
from langchain.llms import OpenAI
from langchain.agents import create_pandas_dataframe_agent
from langchain.chat_models import ChatOpenAI
from langchain.agents.agent_types import AgentType
from langchain.callbacks import StreamlitCallbackHandler
import tabulate

openai.api_key = st.secrets["OPENAI_API_KEY"]

# wide layout
st.set_page_config(page_icon="🤖", page_title="TalkData2Me")

st_callback = StreamlitCallbackHandler(st.container())

footer_html = """
    <div class="footer">
    <style>
        .footer {
            position: fixed;
            bottom: 0;
            left: 0;
            right: 0;
            background-color: #f0f2f6;
            padding: 10px 20px;
            text-align: center;
        }
        .footer a {
            color: #4a4a4a;
            text-decoration: none;
        }
        .footer a:hover {
            color: #3d3d3d;
            text-decoration: underline;
        }
    </style>
        Connect with me on <a href="https://www.linkedin.com/in/chiawei-chien" target="_blank">LinkedIn</a>. 
        If you like this app, consider <a href="https://www.buymeacoffee.com/digitalmagic" target="_blank">buying me a coffee</a> ☕
    </div>
"""

st.title("TalkData2Me 🤖")
st.header('Use LLM to Understand Your Data 🧠')

uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file is None:
    st.info(f"""
                👆 Upload a .csv file first. Sample to try: [sample_data.csv](https://docs.google.com/spreadsheets/d/e/2PACX-1vTeB7_jzJtacH3XrFh553m9ahL0e7IIrTxMhbPtQ8Jmp9gCJKkU624Uk1uMbCEN_-9Sf7ikd1a85wIK/pub?gid=0&single=true&output=csv)
                """)

elif uploaded_file:
    df = pd.read_csv(uploaded_file)

    # Apply the custom function and convert date columns
    for col in df.columns:
        # check if a column name contains date substring
        if 'date' in col.lower():
            df[col] = pd.to_datetime(df[col])
            # remove timestamp
            #df[col] = df[col].dt.date

    # reset index
    df = df.reset_index(drop=True)

    # replace space with _ in column names
    df.columns = df.columns.str.replace(' ', '_')

    cols = df.columns
    cols = ", ".join(cols)

    agent = create_pandas_dataframe_agent(
    ChatOpenAI(temperature=0, model="gpt-3.5-turbo"),
    df,
    verbose=True,
    agent_type=AgentType.OPENAI_FUNCTIONS)

    with st.expander("Preview of the uploaded file"):
        st.table(df.head())
    
    if prompt := st.chat_input():
        st.chat_message("user").write(prompt)
        with st.chat_message("assistant"):
            st_callback = StreamlitCallbackHandler(st.container())
            response = agent.run(prompt, callbacks=[st_callback])
            st.write(response)

# footer
st.markdown(footer_html, unsafe_allow_html=True)



