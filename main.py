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
import sys
import shutil


# Load environment variables
load_dotenv()

if not os.getenv("GROQ_API_KEY"):
    print("GROQ_API_KEY missing in .env file")
    sys.exit(1)


# Step 1: Load PDF safely
print("Loading PDF...")

pdf_path = r'C:\Users\inchu\Downloads\Health.pdf'
pdf_name = os.path.basename(pdf_path)

try:
    reader = PdfReader(pdf_path)

    documents = []

    for page_num, page in enumerate(reader.pages):
        extracted = page.extract_text()

        if extracted:
            documents.append({
                "text": extracted,
                "page": page_num + 1,
                "source": pdf_name
            })

    if not documents:
        print("No text extracted from PDF.")
        sys.exit(1)

    print("PDF loaded successfully")

except Exception as e:
    print(f"PDF Error: {e}")
    sys.exit(1)


# Step 2: Split text and keep metadata
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

print(f"Total Chunks Created: {len(texts)}")


# Step 3: Reset Vector DB
if os.path.exists("chroma_db"):
    shutil.rmtree("chroma_db")

print("Initializing Vector DB...")

embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5"
)

vectordb = Chroma.from_texts(
    texts=texts,
    embedding=embeddings,
    metadatas=metadatas
)

retriever = vectordb.as_retriever(search_kwargs={"k": 3})

print("Vector DB ready")


# Step 4: LLM
llm = ChatGroq(
    model_name="llama-3.3-70b-versatile",
    temperature=0.1
)


# Step 5: Prompt Template
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

print("\nReady! Ask questions.")
print("Type 'exit' to quit.\n")


# Step 6: Q&A Loop
while True:

    question = input("Your Question: ")

    if question.lower() in ["exit", "quit"]:
        print("Bye!")
        break

    docs = retriever.invoke(question)

    if not docs:
        print("Answer: Not enough info in document.")
        continue

    context = "\n\n".join([d.page_content for d in docs])

    print("Answering...")

    answer = chain.invoke({
        "context": context,
        "question": question
    })

    print("\nAnswer:", answer)

    # Print Sources
    sources = set(
        f"{d.metadata['source']} (Page {d.metadata['page']})"
        for d in docs
    )

    print("\nSources:")
    for s in sources:
        print("-", s)