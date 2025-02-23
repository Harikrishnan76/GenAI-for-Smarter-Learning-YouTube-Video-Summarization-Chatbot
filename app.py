import streamlit as st
import validators
from langchain.chains.summarize import load_summarize_chain
from langchain_groq import ChatGroq
from youtube_transcript_api import YouTubeTranscriptApi
from langchain_community.document_loaders import YoutubeLoader
from langchain_core.prompts import PromptTemplate
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import speech_recognition as sr
import os
from dotenv import load_dotenv

load_dotenv()

os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

# Set page config FIRST
st.set_page_config(page_icon="🤖", page_title="Bot For Summarization")

# Remove Gemini gradient background
st.markdown(
    """
    <style>
        .stTextInput>div>div>input {
            background-color: #333;
            color: white;
        }
        .stTextArea>div>div>textarea {
            background-color: #333;
            color: white;
        }
        .stButton>button {
            background: linear-gradient(to right, #6a11cb, #2575fc); /* Button gradient */
            color: white;
            border: none;
            position: relative;
            overflow: hidden;
        }
        .stButton>button::before {
            content: '';
            position: absolute;
            top: 0;
            left: -100%;
            width: 100%;
            height: 100%;
            background: linear-gradient(to right, rgba(255, 255, 255, 0.3), rgba(255, 255, 255, 0.8), rgba(255, 255, 255, 0.3));
            transition: left 0.5s;
        }
        .stButton>button:hover::before {
            left: 100%;
        }
        .stMarkdown h1, h2, h3, h4, h5, h6, .gradient-text { /* Added .gradient-text */
            background: linear-gradient(to right, #6a11cb, #2575fc);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🤖 Bot For Summarization")
st.markdown("<h3 class='gradient-text'>Enter your URL</h3>", unsafe_allow_html=True)
url = st.text_input("URL", label_visibility="collapsed")

template = """
Provide me the summary of the content ,
content:{text}
"""
template2 = """
give me the final summary of the contents with the title for the content,
content:{text}
"""
prompt = PromptTemplate(
    input_variables=["text"],
    template=template,
)
prompt2 = PromptTemplate(input_variables=['text'], template=template2)
llm = ChatGroq(model="mixtral-8x7b-32768")

if 'response' not in st.session_state:
    st.session_state.response = ''

if st.button("Summarize the URL"):
    if not url.strip():
        st.error("Please enter the URL")
    elif not validators.url(url):
        st.error("Please enter a valid URL")
    else:
        with st.spinner("Summarizing..."):
            if "youtube.com" in url:
                loader = YoutubeLoader.from_youtube_url(
                    url, language=['en', 'hi', 'ta', 'te', 'es', 'ru', 'ja', 'de', 'ml']
                )
                documents = loader.load()
                summarize_chain = load_summarize_chain(
                    llm, chain_type="map_reduce", map_prompt=prompt, combine_prompt=prompt2
                )
                st.session_state.response = summarize_chain.run(documents)

if st.session_state.response:
    st.markdown(
        f"""
        <div style="background: linear-gradient(to right, #6a11cb, #2575fc); padding: 15px; border-radius: 5px; color: white;">
            {st.session_state.response}
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("<h3 class='gradient-text'>What's your Question</h3>", unsafe_allow_html=True)
question = st.text_input(label="", placeholder="Enter your Question")

def voice_input():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        st.info("🎙 Listening...")
        audio = recognizer.listen(source)
        try:
            return recognizer.recognize_google(audio)
        except sr.UnknownValueError:
            return "Could not recognize speech."
        except sr.RequestError:
            return "Error connecting to speech recognition service."

if st.button("🎙 Use Voice for Question"):
    question = voice_input()
    st.write(question)

if question:
    prompt3 = ChatPromptTemplate.from_messages(
        [
            ('system', "Answer the user's question based on the provided summary. If the question is unrelated, answer it to the best of your ability."),
            ('user', "Summary: {response}\nQuestion: {question}"),
        ]
    )
    parser = StrOutputParser()
    chain = prompt3 | llm | parser
    answer = chain.invoke(
        {'response': st.session_state.response, 'question': question}
    )
    st.markdown(
        f"""
        <div style="background: linear-gradient(to right, #6a11cb, #2575fc); padding: 15px; border-radius: 5px; color: white;">
            {answer}
        </div>
        """,
        unsafe_allow_html=True,
    )