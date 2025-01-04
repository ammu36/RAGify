import streamlit as st
import faiss
import numpy as np
# from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
from utility.metadata_helpers import MetaDataHelpers
from utility.chunk_helpers import ChunkHelpers
import openai
import os  # Import the os module to check for file existence

# Initialize the embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')
METADATA_FILE = "metadata.json"
dimension = 384  # Embedding size

# Set OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")

# Streamlit App
st.title("RAGify - Where Your PDFs Become Knowledge Wizards!")
st.write("Transform static documents into an interactive Q&A experience powered by AI. Who knew PDFs could talk?")

# Load existing index and metadata (with in-memory clearing)
index = MetaDataHelpers.load_index(dimension)
if os.path.exists(METADATA_FILE):
    metadata = MetaDataHelpers.load_metadata()
else:
    metadata = []

# Initialize session state for storing question-answer pairs
if 'qa_pairs' not in st.session_state:
    st.session_state.qa_pairs = []

# Upload and process PDFs
uploaded_files = st.file_uploader("Upload PDFs", type=["pdf"], accept_multiple_files=True)

if uploaded_files:
    # Reset the FAISS index and metadata
    MetaDataHelpers.reset_index_and_metadata()

    # Create a new FAISS index
    index = faiss.IndexFlatL2(dimension)

    for uploaded_file in uploaded_files:
        # Process the uploaded PDF into chunks
        chunks = ChunkHelpers.process_pdf_to_chunks(uploaded_file)

        # Generate embeddings for each chunk
        embeddings = model.encode(chunks)

        # Add embeddings to the FAISS index
        index.add(np.array(embeddings).astype('float32'))

        # Update metadata with chunk info
        for i, chunk in enumerate(chunks):
            metadata.append({
                "id": len(metadata),
                "file_name": uploaded_file.name,
                "chunk": chunk
            })

    # Save index and metadata locally
    MetaDataHelpers.save_index(index)
    MetaDataHelpers.save_metadata(metadata)

    st.success(f"Processed and added chunks from {len(uploaded_files)} file(s)!")

if st.session_state.qa_pairs:
    st.subheader("Conversation History")
    for qa in st.session_state.qa_pairs:
        st.write(f"**Q:** {qa['question']}")
        st.write(f"**A:** {qa['answer']}")
        st.write("---")

# Query the FAQ system
query = st.text_input("Ask a question:")

if query:
    # Generate embedding for the query
    query_embedding = model.encode([query])

    # Search the FAISS index
    distances, indices = index.search(np.array(query_embedding).astype('float32'), k=10)

    # Retrieve relevant chunks based on search results
    relevant_contexts = [metadata[idx]["chunk"] for idx in indices[0]]

    # Join the retrieved chunks into a single string to form the context for the AI model
    context_string = " ".join(relevant_contexts)

    # st.write("Retrieved Contexts:")
    # for context in relevant_contexts:
    #     st.write(context)
    # st.write(context_string)
    # st.write(query)

    # Use OpenAI to generate a response based on the retrieved context
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are an assistant helping answer questions based on provided contexts. "
                                          "Give an answer in 2-3 sentences."},
            {"role": "user", "content": f"Context: {context_string}\n\nQuestion: {query}"}
        ]
    )

    # Extract and display the AI-generated answer
    answer = response['choices'][0]['message']['content']
    st.write("AI-generated Answer:")
    st.write(answer)
    st.session_state.qa_pairs.append({"question": query, "answer": answer})
