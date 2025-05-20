# Import necessary components from LangChain and Ollama
from langchain_community.document_loaders import PyPDFLoader  # For loading PDF documents
from langchain_text_splitters import RecursiveCharacterTextSplitter  # To split large text into manageable chunks
from langchain.vectorstores import FAISS  # Vector store for similarity search
from langchain_core.vectorstores import InMemoryVectorStore  # (Optional) In-memory vector store for testing
from langchain_ollama import OllamaEmbeddings  # For embedding text using Ollama-compatible models
from langchain_core.prompts import ChatPromptTemplate  # To define the prompt template
from langchain_ollama.llms import OllamaLLM  # To use DeepSeek model via Ollama

# Directory where uploaded PDFs will be saved
pdfs_directory = 'pdfs/'

# Load the DeepSeek LLM model using Ollama for generating responses
embeddings = OllamaEmbeddings(model="deepseek-r1:7b")  # Used for converting text chunks into vector embeddings
model = OllamaLLM(model="deepseek-r1:7b")  # LLM that answers based on context and prompt

# Prompt template used to guide the LLM in generating relevant and concise answers
template = """
You are an assistant that answers questions. Using the following retrieved information, answer the user question. 
If you don't know the answer, say that you don't know. Use up to three sentences, keeping the answer concise.
Question: {question} 
Context: {context} 
Answer:
"""


def upload_pdf(file):
    """
    Saves the uploaded PDF file to the local 'pdfs/' directory.
    """
    with open(pdfs_directory + file.name, "wb") as f:
        f.write(file.getbuffer())


def create_vector_store(file_path):
    """
    Loads a PDF, splits its text into chunks, generates embeddings, 
    and stores them in a FAISS vector store for similarity-based retrieval.
    """
    loader = PyPDFLoader(file_path)
    documents = loader.load()  # Load the text content from the PDF

    # Split the content into chunks for better embedding and search performance
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,        # Max characters per chunk
        chunk_overlap=300,      # Overlap between chunks to preserve context
        add_start_index=True    # Keep track of chunk positions
    )

    chunked_docs = text_splitter.split_documents(documents)

    # Create a FAISS vector store from the embedded document chunks
    db = FAISS.from_documents(chunked_docs, embeddings)
    return db


def retrieve_docs(db, query, k=4):
    """
    Performs a similarity search in the vector store to retrieve top-k relevant chunks.
    """
    print(db.similarity_search(query))  # For debugging/logging
    return db.similarity_search(query, k)


def question_pdf(question, documents):
    """
    Generates a response to the user's question using the retrieved documents 
    and the DeepSeek LLM guided by the prompt template.
    """
    # Combine all retrieved documents into a single context string
    context = "\n\n".join([doc.page_content for doc in documents])

    # Format the question and context into the pre-defined prompt
    prompt = ChatPromptTemplate.from_template(template)

    # Pipe the prompt into the model to generate a response
    chain = prompt | model
    return chain.invoke({"question": question, "context": context})
