# 📄 Legal Document Assistant (RAG Chatbot)

A **Retrieval-Augmented Generation (RAG) chatbot** that answers questions from legal documents using AI.

The system processes PDF documents, creates embeddings, stores them in a vector database, and retrieves relevant context to generate accurate answers using a Large Language Model (LLM).

---

## 🚀 Features

* Ask questions about legal documents
* Context-aware answers using **RAG architecture**
* PDF document processing
* Semantic search using **vector embeddings**
* Fast retrieval using **Chroma Vector Database**
* Powered by **Groq LLM**
* Interactive UI built with **Streamlit**

---

## 🧠 Tech Stack

* Python
* Streamlit
* LangChain
* Hugging Face Embeddings
* Chroma Vector Database
* Groq LLM
* PyPDF2

---

## 🏗️ Project Architecture

PDF Document
⬇
Text Extraction
⬇
Text Chunking
⬇
Embeddings Generation
⬇
Vector Database (Chroma)
⬇
Retriever
⬇
LLM (Groq)
⬇
Answer Generation

---

## 📂 Project Structure

```
legal-rag-botzz
│
├── app.py
├── requirements.txt
├── Health.pdf
├── chroma_db/
└── README.md
```

---

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/yourusername/legal-rag-botzz.git
cd legal-rag-botzz
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate the environment (Windows):

```bash
venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## 🔑 Setup API Key

Create a `.env` file and add your Groq API key:

```
GROQ_API_KEY=your_groq_api_key
```

For deployment on Streamlit Cloud, add the key in **Streamlit Secrets**.

---

## ▶️ Run the Application

```bash
streamlit run app.py
```

---

## 🌐 Deployment

This project can be deployed using:

* Streamlit Community Cloud

---

## 💡 Example Questions

* What is the maximum ICU room rent limit per day as a percentage of the sum insured?
* What is the maximum cap for cataract treatment per eye?
* After how many claim-free years is a free medical check-up allowed?

---

## 👩‍💻 Author

Adithi Inchure
