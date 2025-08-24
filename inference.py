import streamlit as st
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain_postgres import PGVector
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
import os

# Streamlit page configuration
st.set_page_config(
    page_title="Trade Finance Compliance Assistant",
    page_icon="📋",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title and description
st.title("📋 Trade Finance Compliance Assistant")
st.markdown("Ask questions about trade finance compliance and get answers based on UCP 600, ISBP 821, URC 522, URDG 758, and SWIFT MT7xx standards.")

# Sidebar for configuration
st.sidebar.header("Configuration")

# Configuration settings
COLLECTION_NAME = "doccredit"
CONNECTION_STRING = "postgresql+psycopg://shahul:Cpu%4012345@shahulhannah.postgres.database.azure.com:5432/postgres"

# Initialize session state
if 'vectorstore' not in st.session_state:
    st.session_state.vectorstore = None
if 'rag_chain' not in st.session_state:
    st.session_state.rag_chain = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

def initialize_embeddings():
    """Initialize Azure OpenAI embeddings"""
    embeddings = AzureOpenAIEmbeddings(
        azure_endpoint="https://shace-mejiegxb-eastus2.cognitiveservices.azure.com/",
        api_key="9LTf4DOFBLDVHcCm9LaEcJ1hLUK3QIhzDVqGOhKqh6nsF3VBIvoLJQQJ99BHACHYHv6XJ3w3AAAAACOG77Gk", 
        openai_api_version="2024-02-01",
        azure_deployment="text-embedding-3-large",
        model="text-embedding-3-large",
    )
    return embeddings

def initialize_vectorstore():
    """Initialize PGVector vector store"""
    embeddings = initialize_embeddings()
    vectorstore = PGVector(
        connection=CONNECTION_STRING,
        embeddings=embeddings,
        collection_name=COLLECTION_NAME,
        use_jsonb=True,
    )
    return vectorstore

def initialize_llm():
    """Initialize Azure OpenAI chat model"""
    llm = AzureChatOpenAI(
        azure_endpoint="https://shace-mejiegxb-eastus2.cognitiveservices.azure.com/",
        api_key="9LTf4DOFBLDVHcCm9LaEcJ1hLUK3QIhzDVqGOhKqh6nsF3VBIvoLJQQJ99BHACHYHv6XJ3w3AAAAACOG77Gk",
        openai_api_version="2024-12-01-preview",
        azure_deployment="gpt-4o",
        model="gpt-4o",
    )
    return llm

def initialize_rag_chain():
    """Initialize the RAG chain"""
    vectorstore = initialize_vectorstore()
    llm = initialize_llm()
    
    system_prompt = """
You are a Trade Finance Compliance Assistant.
    Answer ONLY from the provided context snippets (verbatim extracts from standards like UCP 600, ISBP 821, URC 522, URDG 758, SWIFT MT7xx, bank policies)
    - Cite rule identifiers precisely when visible 
    - If the context is insufficient, reply exactly: 'Insufficient evidence in the provided sources.
    - Keep answers concise and structured; 
    - ***Find all the discrpancies with reasoning as per the below format requirements***
    Expected Output Template should be in this format :::: 
    FORMAT REQUIREMENTS:
If sufficient evidence exists in the provided sources, respond ONLY in the following exact layout (no extra text before or after):
Overall Decision. (Compliant | Not Compliant)
Discrepancies.
Discrepancy 1. (short description of the issue) 
Discrepancy 1 (Detailed description of the issue)
Against (Standard / Article No.)
Required 1. (what is needed to cure)
Criticality 1 (High | Medium | Low)

Discrepancy 2. (short description)
Against (Standard / Article No.)
Required 2. (what is needed to cure)
Criticality 2 (High | Medium | Low)
[Add Discrepancy 3+ blocks as needed. If none, write: "Discrepancies. None."]

Rules:
- Use only verbatim facts from the provided context.
- Do not include any content outside the template.
- If evidence is insufficient, DO NOT use the template; reply exactly: "Insufficient evidence in the provided source
    {context}
    
    """

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}"),
        ]
    )
    
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(vectorstore.as_retriever(), question_answer_chain)
    
    return rag_chain

def ingest_documents():
    """Function to ingest documents (only run when needed)"""
    st.sidebar.subheader("Document Ingestion")
    
    uploaded_file = st.sidebar.file_uploader("Upload corpus.txt file", type=['txt'])
    
    if uploaded_file is not None:
        if st.sidebar.button("Ingest Documents"):
            with st.spinner("Ingesting documents..."):
                # Read the uploaded file
                corpus_text = uploaded_file.read().decode('utf-8')
                
                # Convert to LangChain Document
                docs = [Document(page_content=corpus_text)]
                
                # Chunk the document
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                chunked_docs = text_splitter.split_documents(docs)
                
                # Initialize vectorstore and add documents
                vectorstore = initialize_vectorstore()
                vectorstore.add_documents(chunked_docs)
                
                st.sidebar.success(f"Successfully ingested {len(chunked_docs)} document chunks!")

# Sidebar for document ingestion
ingest_documents()

# Initialize the RAG chain
try:
    if st.session_state.rag_chain is None:
        with st.spinner("Initializing Trade Finance Compliance Assistant..."):
            st.session_state.rag_chain = initialize_rag_chain()
        st.success("✅ Assistant initialized successfully!")
except Exception as e:
    st.error(f"❌ Error initializing assistant: {str(e)}")
    st.stop()

# Main chat interface
st.header("💬 Ask Your Question")

# Example questions


# Chat input
user_question = st.text_area(
    "Enter your trade finance compliance question:",
    height=150,
    placeholder="Paste your documentary credit details, Bill of Lading information, or any trade finance compliance question here..."
)


# Submit button
col1, col2, col3 = st.columns([1, 1, 4])
with col1:
    if st.button("🔍 Analyze", type="primary"):
        if user_question.strip():
            with st.spinner("Analyzing your question..."):
                try:
                    # Get response from RAG chain
                    response = st.session_state.rag_chain.invoke({"input": user_question})
                    
                    # Add to chat history
                    st.session_state.chat_history.append({
                        "question": user_question,
                        "answer": response["answer"],
                        "sources": response["context"]
                    })
                    
                    # Display the response
                    st.header("📋 Analysis Result")
                    st.markdown(response["answer"])
                    
                    # Display sources
                    with st.expander("📚 Source Documents"):
                        for i, doc in enumerate(response["context"], 1):
                            st.markdown(f"**Source {i}:**")
                            st.text(doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content)
                            st.markdown("---")
                    
                except Exception as e:
                    st.error(f"❌ Error processing your question: {str(e)}")
        else:
            st.warning("⚠️ Please enter a question before analyzing.")
with col2:
    if st.button("Back"):
        if len(st.session_state.chat_history) > 0:
            st.session_state.chat_history.pop()
            st.rerun()
with col3:
    if st.session_state.chat_history:
        latest_chat = st.session_state.chat_history[-1]
        combined_text = f"Question:\n{latest_chat["question"]}\n\nAnswer:\n{latest_chat["answer"]}"
        st.download_button(
            label="Copy Question and Answer",
            data=combined_text,
            file_name="question_answer.txt",
            mime="text/plain"
        )

# Chat history
if st.session_state.chat_history:
    st.header("📜 Chat History")
    
    for i, chat in enumerate(reversed(st.session_state.chat_history)):
        with st.expander(f"Question {len(st.session_state.chat_history) - i}: {chat['question'][:100]}..."):
            st.markdown("**Question:**")
            st.text(chat['question'])
            st.markdown("**Answer:**")
            st.markdown(chat['answer'])

# Clear chat history button
if st.session_state.chat_history:
    if st.button("🗑️ Clear Chat History"):
        st.session_state.chat_history = []
        st.rerun()

# Footer
st.markdown("---")
st.markdown("*Trade Finance Compliance Assistant - Powered by Azure OpenAI and LangChain*")

