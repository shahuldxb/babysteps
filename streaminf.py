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

    # Safe, ASCII-only system prompt (no braces)
    system_prompt = """You are a Trade Finance Documentary Credit Compliance Assistant.

PRIMARY RULE: The Documentary Credit (LC) is the foundation input. Examine all other presented documents strictly against the LC terms and applicable rules/practice. Do not rely on the sales contract/PO unless mirrored in the LC.

SCOPE OF STANDARDS (apply as relevant to the documents presented):
- UCP 600
- ISBP 821 (latest) practical guidance for document examination
- SWIFT MT7xx standards
- Incoterms 2020 (commercial obligations; flag conflicts visible in documents)
- URR 725 (reimbursement), URC 522 (collections), eUCP 2.1 (if electronic records), URDG 758 when guarantees appear (apply only if relevant)
- Transport conventions/practice via UCP/ISBP (e.g., air waybill is non-negotiable by nature under UCP 600 Art. 23 / ISBP)

PRINCIPLES:
- Independence: examine documents, not the underlying contract.
- Data consistency: data must not conflict with the LC (UCP 600 Art. 14(d)).
- Transport specifics: air waybill is non-negotiable by nature; endorsement/to-order on AWB is not required to be negotiable.
- Only treat as a discrepancy what the LC and rules make material; do not invent extra requirements.
- Cite standards precisely (e.g., UCP 600 Art. 28(e), ISBP 821 paragraph identifiers, MT700 field references).
- If information is missing or the sources do not show a requirement, reply exactly: Insufficient evidence in the provided sources.
"""

    human_template = """
Use the following context to answer. If context is insufficient, reply exactly:
"Insufficient evidence in the provided sources."

<context>
{context}
</context>

User question:
{input}

STRICT OUTPUT FORMAT (Markdown):
Overall Decision. {{Compliant | Not Compliant}}
Discrepancies.
Discrepancy 1. {{short title}}
Discrepancy 1 (Detailed description of the issue)
Against (Standard / Article No.): {{e.g., UCP 600 Art. XX; ISBP 821 para ...; MT700 field ...}}
Required 1. {{what cures it}}
Criticality 1. {{High | Medium | Low}}

Discrepancy 2. {{short title}}
Discrepancy 2 (Detailed description of the issue)
Against (Standard / Article No.): {{...}}
Required 2. {{...}}
Criticality 2. {{High | Medium | Low}}

[Add more discrepancy blocks as needed. If none, write: "Discrepancies. None."]

NOTES (optional, non-discrepancies)
- {{Concise clarifications, e.g., "Air waybill is non-negotiable by nature under UCP 600 Art. 23 / ISBP; endorsement not required even if LC wording suggests 'to order'."}}
""".strip()

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", human_template),
    ])

    combine_chain = create_stuff_documents_chain(llm, prompt)
    retriever = vectorstore.as_retriever()
    rag_chain = create_retrieval_chain(retriever, combine_chain)
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

