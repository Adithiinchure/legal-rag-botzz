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
import os
import sys
import shutil


# Load environment variables


load_dotenv()


if not os.getenv("GROQ_API_KEY"):
   print("GROQ_API_KEY missing in .env file")
   sys.exit(1)


# Step 1: Load PDF safely


print("Loading PDF...")


try:
   reader = PdfReader(r'C:\Users\inchu\Downloads\Health.pdf')
   pages_text = []
   for page in reader.pages:
       extracted = page.extract_text()
       if extracted:                     # Avoid None values
           pages_text.append(extracted)


   text = "\n".join(pages_text)


   if not text.strip():
       print("No text extracted from PDF.")
       sys.exit(1)


   print("PDF loaded successfully")
   print("Total text length:", len(text))


except Exception as e:
   print(f"PDF Error: {e}")
   sys.exit(1)




# Step 2: Split text


splitter = RecursiveCharacterTextSplitter(
   chunk_size=1000,
   chunk_overlap=200
)


chunks = splitter.split_text(text)


if len(chunks) == 0:
   print("Chunking failed: No chunks created.")
   sys.exit(1)


print(f"Total Chunks Created: {len(chunks)}")




# Step 3: Reset Vector DB (important!)


if os.path.exists("chroma_db"):
   shutil.rmtree("chroma_db")


print("Initializing Vector DB...")


embeddings = HuggingFaceEmbeddings(
   model_name="BAAI/bge-small-en-v1.5"
)


vectordb = Chroma.from_texts(
   texts=chunks,
   embedding=embeddings,
   persist_directory="chroma_db"
)


retriever = vectordb.as_retriever(search_kwargs={"k": 5})


print("Vector DB ready")




# Step 4: LLM


llm = ChatGroq(
   model_name="llama-3.1-8b-instant",
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


   print("Answer:", answer)