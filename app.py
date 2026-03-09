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

load_dotenv()

st.set_page_config(page_title="Legal RAG Bot", layout="wide")

# -------- TITLE --------
st.title("⚖️ Legal Document RAG Chatbot")
st.write("Upload a legal PDF and ask questions based only on the document.")

# -------- CHAT HISTORY INIT --------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

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

    documents = []
    pdf_name = uploaded_file.name

    for page_num, page in enumerate(reader.pages):
        text = page.extract_text()

        if text:
            documents.append({
                "text": text,
                "page": page_num + 1,
                "source": pdf_name
            })

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    texts = []
    metadatas = []

    for doc in documents:
        chunks = splitter.split_text(doc["text"])

        for chunk in chunks:
            texts.append(chunk)
            metadatas.append({
                "page": doc["page"],
                "source": doc["source"]
            })

    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en-v1.5"
    )

    vectordb = Chroma.from_texts(
        texts=texts,
        embedding=embeddings,
        metadatas=metadatas
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

    # Sidebar chat history
    st.sidebar.markdown("### 💬 Chat History")

    for msg in st.session_state.chat_history:

        if msg["role"] == "user":
            st.sidebar.markdown(f"**Question:** {msg['content']}")

        else:
            st.sidebar.markdown(f"**Answer:** {msg['content']}")

            if "sources" in msg:
                st.sidebar.markdown("Sources:")
                for s in msg["sources"]:
                    st.sidebar.write(f"📄 {s}")

        st.sidebar.markdown("---")

    # Main page input
    user_question = st.chat_input("Ask a question about the document...")

    if user_question:

        # Show question in main page
        st.chat_message("user").write(user_question)

        docs = st.session_state.retriever.invoke(user_question)

        context = "\n\n".join([d.page_content for d in docs])

        with st.spinner("Thinking..."):
            answer = st.session_state.chain.invoke({
                "context": context,
                "question": user_question
            })

        # Show answer in main page
        with st.chat_message("assistant"):
            st.write(answer)

            st.markdown("**Sources:**")

            source_list = []
            for d in docs:
                source = f"{d.metadata.get('source')} — Page {d.metadata.get('page')}"
                source_list.append(source)
                st.write(f"📄 {source}")

        # Store in sidebar history
        st.session_state.chat_history.append({
            "role": "user",
            "content": user_question
        })

        st.session_state.chat_history.append({
            "role": "assistant",
            "content": answer,
            "sources": source_list
        })

else:
    st.info("Upload and process a document to start chatting.")

