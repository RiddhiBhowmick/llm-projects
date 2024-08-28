import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_google_genai import ChatGoogleGenerativeAI  # Import for Gemini
import os
from dotenv import load_dotenv
import google.generativeai as genai
from langchain_google_genai import GoogleGenerativeAIEmbeddings,ChatGoogleGenerativeAI

# Load environment variables
load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# os.environ["GOOGLE_API_KEY"] = "your_google_api_key"  # Replace with your Google API key

def pdf_read(pdf_docs):
    """Read PDF files and extract their text content."""
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
    return text

def get_chunks(text):
    """Split the text into manageable chunks."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_text(text)
    return chunks

def create_vector_store(chunks):
    """Create a FAISS vector store from text chunks."""
    # embeddings = OpenAIEmbeddings()  # Use OpenAI embeddings for chunking
    embeddings=GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(chunks, embedding=embeddings)
    vector_store.save_local("faiss_db")  # Save the vector store locally
    return vector_store

def load_vector_store():
    """Load the FAISS vector store from disk."""
    # embeddings = OpenAIEmbeddings()
    embeddings=GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    return FAISS.load_local("faiss_db", embeddings, allow_dangerous_deserialization=True)

def main():
    st.set_page_config(page_title="Chat with PDF")
    st.header("Chat with Your PDF Documents")

    # File uploader for PDF documents
    pdf_docs = st.file_uploader("Upload PDF files", type=["pdf"], accept_multiple_files=True)

    if st.button("Submit & Process"):
        if pdf_docs:
            with st.spinner("Processing..."):
                text = pdf_read(pdf_docs)
                chunks = get_chunks(text)
                create_vector_store(chunks)
                st.success("PDFs processed and vector store created!")

    # Input for user question
    user_question = st.text_input("Ask a question about the PDF content:")
    
    if user_question:
        with st.spinner("Searching for answer..."):
            vector_store = load_vector_store()
            llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro")  # Use Gemini model
            retriever = vector_store.as_retriever()
            qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                retriever=retriever
            )

            result = qa_chain.run(user_question)
            st.write("Answer:", result)

if __name__ == "__main__":
    main()