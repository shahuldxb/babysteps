from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain_postgres import PGVector
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from tqdm import tqdm
# Step 1: Load the corpus.txt file
with open("corpus.txt", "r", encoding="utf-8") as f:
    corpus_text = f.read()
print("Step 1: Loaded corpus.txt, length:", len(corpus_text), "characters")

# Convert to a LangChain Document object
docs = [Document(page_content=corpus_text)]
print("Step 1: Converted to", len(docs), "LangChain Document object(s)")

# Step 2: Chunk the document
text_splitter = RecursiveCharacterTextSplitter(chunk_size=32000, chunk_overlap=3200)
chunked_docs = text_splitter.split_documents(docs)
print("Step 2: Split into", len(chunked_docs), "chunks")

# Step 3: Set up Azure OpenAI embeddings
embeddings = AzureOpenAIEmbeddings(
    azure_endpoint= "https://shace-mejiegxb-eastus2.cognitiveservices.azure.com/",
    api_key= "9LTf4DOFBLDVHcCm9LaEcJ1hLUK3QIhzDVqGOhKqh6nsF3VBIvoLJQQJ99BHACHYHv6XJ3w3AAAAACOG77Gk", 
    openai_api_version="2024-02-01",
    azure_deployment="text-embedding-3-large",
    model="text-embedding-3-large",
)
print("Step 3: Initialized Azure OpenAI embeddings")

# Step 4: Create PGVector vector store
CONNECTION_STRING = "postgresql+psycopg://shahul:Cpu%4012345@shahulhannah.postgres.database.azure.com:5432/postgres"
##postgresql+psycopg://shahul:Cpu%4012345@shahulhannah.postgres.database.azure.com:5432/postgres

vectorstore = PGVector(
    connection=CONNECTION_STRING,
    embeddings=embeddings,  # Note: Use 'embedding_function' as per some docs; if error, try 'embeddings'
    collection_name="doccredit",
    use_jsonb=True
)
print("Step 4: Initialized PGVector vector store")
# Add the chunked documents to the vector store with a progress bar
for doc in tqdm(chunked_docs, desc="Adding documents to PGVector", unit="doc"):
    vectorstore.add_documents([doc])  # Add one document at a time to show progress
print("Step 4: Added", len(chunked_docs), "documents to PGVector")


# Step 5: Set up the Azure OpenAI chat model for RAG
llm = AzureChatOpenAI(
    azure_endpoint="https://shace-mejiegxb-eastus2.cognitiveservices.azure.com/",
    api_key= "9LTf4DOFBLDVHcCm9LaEcJ1hLUK3QIhzDVqGOhKqh6nsF3VBIvoLJQQJ99BHACHYHv6XJ3w3AAAAACOG77Gk",
    openai_api_version="2024-12-01-preview",
    azure_deployment="gpt-4o",
    model="gpt-4o",
)
print("Step 5: Initialized Azure OpenAI chat model (gpt-4o)")

# Step 6: Set up the RAG chain
system_prompt = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer the question. "
    "If you don't know the answer, say that you don't know. "
    "Use three sentences maximum and keep the answer concise."
    "\n\n"
    "{context}"
)
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)
print("Step 6: Created ChatPromptTemplate for RAG")

# Create the document stuffing chain and retrieval chain
question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(vectorstore.as_retriever(), question_answer_chain)
print("Step 6: Initialized RAG chain with retriever and QA chain")

# Step 7: Example usage - Query the RAG chain
query = "What is the main topic in the corpus?"
print("Step 7: Executing query:", query)
response = rag_chain.invoke({"input": query})

# Print the answer and sources
print("Step 7: Answer:", response["answer"])
print("Step 7: Sources (retrieved chunks):")
for i, doc in enumerate(response["context"], 1):
    print(f"  Chunk {i}: {doc.page_content[:100]}...")  # Show first 100 chars of each chunk
