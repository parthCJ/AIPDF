# Import Streamlit for building the web UI
import streamlit as st

# Import the main module which contains core logic functions
import main as main

# Set the title of the Streamlit app
st.title("Chat with PDFs using Deepseek")

# Create a file uploader widget that accepts only one PDF file
uploaded_file = st.file_uploader(
    "Upload PDF",               # Label shown to the user
    type="pdf",                 # Accept only PDF files
    accept_multiple_files=False # Only one file at a time
)

# If a PDF file has been uploaded
if uploaded_file:
    # Save the uploaded file to the server directory
    main.upload_pdf(uploaded_file)

    # Process the PDF: create a vector store from the PDF content
    db = main.create_vector_store(main.pdfs_directory + uploaded_file.name)

    # Provide a chat input field for the user to ask questions
    question = st.chat_input()

    # If the user asks a question
    if question:
        # Display the user's question in the chat interface
        st.chat_message("user").write(question)

        # Retrieve the most relevant chunks from the PDF based on the question
        related_documents = main.retrieve_docs(db, question)

        # Generate a concise answer using the DeepSeek language model
        answer = main.question_pdf(question, related_documents)

        # Display the model's response in the assistant's chat message
        st.chat_message("assistant").write(answer)

        # Debugging line (optional): show raw retrieval results
        # st.text(main.retrieve_docs(db=db, query="NVIDIA STOCK"))
