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

st.title("📄 Legal Document Assistant")

try:
    api_key = st.secrets["GROQ_API_KEY"]
except:
    api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    st.error("GROQ_API_KEY missing.")
    st.stop()

os.environ["GROQ_API_KEY"] = api_key


# ---------- Load PDF and create vector DB ----------
@st.cache_resource
def setup_rag():

    reader = PdfReader("Health.pdf")

    pages_text = []
    for page in reader.pages:
        extracted = page.extract_text()
        if extracted:
            pages_text.append(extracted)

    text = "\n".join(pages_text)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = splitter.split_text(text)

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

    # reduced chunks to avoid rate limits
    retriever = vectordb.as_retriever(search_kwargs={"k": 3})

    # smaller model (fewer rate limits)
    llm = ChatGroq(
        model_name="llama-3.1-8b-instant",
        temperature=0.1
    )

    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="""
You are a legal document assistant.

Answer strictly using only the context provided.
If the answer is not clearly present, say:
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


retriever, chain = setup_rag()


# ---------- UI ----------

question = st.text_input("Ask a question about the document")

if st.button("Get Answer"):

    if question:

        docs = retriever.invoke(question)

        if not docs:
            st.warning("Not enough info in document.")
        else:

            context = "\n\n".join([d.page_content for d in docs])

            with st.spinner("Answering..."):

                try:
                    answer = chain.invoke({
                        "context": context,
                        "question": question
                    })
                    st.success(answer)

                except Exception as e:

                    if "RateLimitError" in str(e):
                        st.error("⚠️ Groq API rate limit reached. Please wait 1–2 minutes and try again.")

                    else:
                        st.error(f"⚠️ Error: {str(e)}")