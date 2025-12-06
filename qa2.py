import streamlit as st
from dotenv import load_dotenv
import os
from io import BytesIO
import PyPDF2
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnableLambda, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

st.title("Research Paper Summarizer")

if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None
if 'processed' not in st.session_state:
    st.session_state.processed = False

col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Upload PDF")
    
    uploaded_file = st.file_uploader("Choose a PDF file", type=['pdf'])
    
    if uploaded_file and st.button("Process PDF"):
        with st.spinner("Processing..."):
            try:
                reader = PyPDF2.PdfReader(BytesIO(uploaded_file.read()))
                text = ""
                for page in reader.pages:
                    extracted = page.extract_text()
                    if extracted:
                        text += extracted
                
                if not text.strip():
                    st.error("Could not extract text from PDF")
                else:
                    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
                    chunks = splitter.split_text(text)
                    
                    if not chunks:
                        st.error("No content to process")
                    else:
                        docs = [Document(page_content=chunk) for chunk in chunks]
                        
                        embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
                        st.session_state.vector_store = FAISS.from_documents(docs, embedding=embeddings)
                        st.session_state.processed = True
                        st.success(f"PDF processed")
                        st.info(f"{len(chunks)} chunks created")
            except Exception as e:
                st.error(f"Error: {str(e)}")
    
    if st.session_state.processed:
        st.markdown("---")
        st.write("Status: Ready")

with col2:
    st.subheader("Ask Questions")
    
    if not st.session_state.processed:
        st.info("Upload and process a PDF first")
    else:
        if st.button("Generate Summary"):
            with st.spinner("Generating summary..."):
                try:
                    retriever = st.session_state.vector_store.as_retriever(search_kwargs={'k': 20})
                    docs = retriever.invoke("main findings methodology results conclusions")
                    context = "\n\n".join([doc.page_content for doc in docs])
                    
                    prompt = PromptTemplate.from_template("""Summarize this research paper in 4-5 sentences:

Context: {context}

Summary:""")
                    
                    model = ChatGoogleGenerativeAI(model="gemini-2.0-flash-exp", temperature=0.3)
                    chain = prompt | model | StrOutputParser()
                    summary = chain.invoke({"context": context})
                    
                    st.write("**Summary:**")
                    st.write(summary)
                except Exception as e:
                    st.error(f"Error: {str(e)}")
        
        st.markdown("---")
        
        question = st.text_input("Ask a question about the paper:")
        
        if st.button("Get Answer") and question:
            with st.spinner("Finding answer..."):
                try:
                    retriever = st.session_state.vector_store.as_retriever(search_kwargs={'k': 10})
                    
                    prompt = PromptTemplate.from_template("""Based on the context, answer the question.

Context: {context}

Question: {question}

Answer:""")
                    
                    model = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.3)
                    
                    def get_context(docs):
                        return "\n\n".join([doc.page_content for doc in docs])
                    
                    chain = RunnableParallel({
                        "context": retriever | RunnableLambda(get_context),
                        "question": RunnablePassthrough()
                    }) | prompt | model | StrOutputParser()
                    
                    answer = chain.invoke(question)
                    
                    st.write("**Answer:**")
                    st.write(answer)
                except Exception as e:
                    st.error(f"Error: {str(e)}")