import streamlit as st
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import shutil

load_dotenv()

st.set_page_config(page_title="Legal RAG Bot", layout="wide")

# -------- TITLE --------
st.title("⚖️ Legal Document RAG Chatbot")
st.write("Upload a legal PDF and ask questions based only on the document.")

# -------- SIDEBAR --------
st.sidebar.header("📂 Upload Document")

uploaded_file = st.sidebar.file_uploader(
    "Choose a PDF file",
    type=["pdf"]
)

process_btn = st.sidebar.button("Process Document")


# ---------- API KEY ----------
try:
    api_key = st.secrets["GROQ_API_KEY"]
except:
    api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    st.error("GROQ_API_KEY missing.")
    st.stop()

os.environ["GROQ_API_KEY"] = api_key


# ---------- RAG Setup ----------
@st.cache_resource
def setup_rag(uploaded_file):

    reader = PdfReader(uploaded_file)

    pages_text = []
    for page in reader.pages:
        text = page.extract_text()
        if text:
            pages_text.append(text)

    full_text = "\n".join(pages_text)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = splitter.split_text(full_text)

    if os.path.exists("chroma_db"):
        shutil.rmtree("chroma_db")

    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en-v1.5"
    )

    vectordb = Chroma.from_texts(
        texts=chunks,
        embedding=embeddings,
        persist_directory="chroma_db"
    )

    retriever = vectordb.as_retriever(search_kwargs={"k": 3})

    llm = ChatGroq(
        model_name="llama-3.1-8b-instant",
        temperature=0.1
    )

    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="""
You are a legal assistant.

Answer only from the provided context.

If the answer is not present say:
Not enough info in document.

Context:
{context}

Question:
{question}

Answer:
"""
    )

    chain = prompt | llm | StrOutputParser()

    return retriever, chain


# ---------- PROCESS DOCUMENT ----------
if process_btn:

    if uploaded_file is None:
        st.warning("Please upload a PDF first.")
    else:
        retriever, chain = setup_rag(uploaded_file)

        st.session_state.retriever = retriever
        st.session_state.chain = chain

        st.success("✅ Document processed successfully!")


# ---------- CHAT ----------
if "retriever" in st.session_state:

    user_question = st.chat_input("Ask a question about the document...")

    if user_question:

        docs = st.session_state.retriever.invoke(user_question)

        context = "\n\n".join([d.page_content for d in docs])

        with st.spinner("Thinking..."):

            answer = st.session_state.chain.invoke({
                "context": context,
                "question": user_question
            })

        st.chat_message("user").write(user_question)
        st.chat_message("assistant").write(answer)

else:
    st.info("Upload and process a document to start chatting.")