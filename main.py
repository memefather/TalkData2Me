import streamlit as st
import pandas as pd
import openai
from langchain.llms import OpenAI
from langchain.agents import create_pandas_dataframe_agent
from langchain.chat_models import ChatOpenAI
from langchain.agents.agent_types import AgentType
from langchain.callbacks import StreamlitCallbackHandler
import tabulate
import matplotlib
from audio_recorder_streamlit import audio_recorder

openai.api_key = st.secrets["OPENAI_API_KEY"]

# wide layout
st.set_page_config(page_icon="🤖", page_title="TalkData2Me")

st.markdown(
    """
    <style>
.css-1jc7ptx, .e1ewe7hr3, .viewerBadge_container__1QSob, .styles_viewerBadge__1yB5_, .viewerBadge_link__1S137, .viewerBadge_text__1JaDK{ display: none; } #MainMenu{ visibility: hidden; } footer { visibility: hidden; } 
    </style>
    """,
    unsafe_allow_html=True
)

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

page_bg_img = f"""
<style>
[data-testid="stAppViewContainer"] > .main {{
background-image: url("https://i.postimg.cc/Px6t2qSn/Zz0z-ZTli-Mj-Q4-Mzhl-NGEx-MWVi-Ym-Ji-Mj-Fi-ZTI2-ZWNm-N2-Mz-ZA.jpg");
background-size: cover;
background-position: center center;
background-repeat: no-repeat;
background-attachment: local;
}}
[data-testid="stHeader"] {{
background: rgba(0,0,0,0);
}}
</style>
"""

st.markdown(page_bg_img, unsafe_allow_html=True)

st.title("TalkData2Me 🤖")
st.header('🧠 Use LLM to Understand Your Data 🦾')

uploaded_file = st.file_uploader("Upload your data", type=["csv"])

if uploaded_file is None:
    st.info(f"""
                👆 Upload a .csv file here. A sample to get you started: [sample_data.csv](https://drive.google.com/uc?export=download&id=10ENm4nZFZrUGnLmv_Esph2SXBRmF0ZNS)
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
    
    prompt = st.text_area("Talk to your data! Clearly describe your question in simple terms.")
    

    audio_bytes = audio_recorder()
    if audio_bytes:
        st.audio(audio_bytes, format="audio/wav")
    
    if prompt != None and st.button('Ask!'):
        st.chat_message("user").write(prompt)
        with st.chat_message("assistant"):
            st_callback = StreamlitCallbackHandler(st.container())
            try:
                response = agent.run(prompt, callbacks=[st_callback])
                st.write(response)
            except:
                st.markdown('Clarify your question and try again!')

# footer
st.markdown(footer_html, unsafe_allow_html=True)



