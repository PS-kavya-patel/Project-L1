import streamlit as st
import os
import pypdf
import chromadb
from chromadb.utils import embedding_functions
from groq import Groq

# Streamlit UI Setup
st.set_page_config(page_title="Ultra-Simple SDS AI", page_icon="🧪")
st.title("SDSense AI – Minimalist RAG")
st.write("A Python implementation of RAG using ChromaDB for semantic search and Groq for fast AI answers!")

# Sidebar for API key and File Upload
with st.sidebar:
    st.header("Settings")
    api_key = st.text_input("Groq API Key", type="password")
    st.markdown("[Get your free Groq API key here](https://console.groq.com/keys)")
    
    st.header("Document Upload")
    uploaded_file = st.file_uploader("Upload SDS PDF", type=["pdf"])
    process_btn = st.button("Process Document")

# Configure Groq Client if key is provided
groq_client = None
if api_key:
    groq_client = Groq(api_key=api_key)

# Initialize Session State
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "chroma_client" not in st.session_state:
    st.session_state.chroma_client = chromadb.Client()
    
# Multilingual Embedding Function
multilingual_ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="paraphrase-multilingual-MiniLM-L12-v2")

if "collection" not in st.session_state:
    # Delete if exists to allow re-processing easily in the same session
    try:
        st.session_state.chroma_client.delete_collection("sds_docs")
    except Exception:
        pass
    st.session_state.collection = st.session_state.chroma_client.create_collection("sds_docs", embedding_function=multilingual_ef)
if "document_processed" not in st.session_state:
    st.session_state.document_processed = False

def extract_text_and_chunk(file, chunk_size=1000, overlap=200):
    """Read text from a PDF file, chunk it, and associate with page numbers."""
    reader = pypdf.PdfReader(file)
    chunks = []
    metadatas = []
    ids = []
    chunk_id = 0
    
    for i, page in enumerate(reader.pages):
        page_text = page.extract_text()
        if page_text:
            start = 0
            while start < len(page_text):
                end = start + chunk_size
                chunk = page_text[start:end]
                chunks.append(chunk)
                metadatas.append({"page": i + 1, "source": file.name})
                ids.append(f"chunk_{chunk_id}")
                chunk_id += 1
                start += chunk_size - overlap
                
    return chunks, metadatas, ids

# Handle Document Processing
if process_btn:
    if not api_key:
        st.error("Please enter your Groq API Key first.")
    elif not uploaded_file:
        st.error("Please upload a PDF document.")
    else:
        with st.spinner("Extracting and chunking text from PDF..."):
            chunks, metadatas, ids = extract_text_and_chunk(uploaded_file)
            
        with st.spinner("Building vector search index (ChromaDB)..."):
            try:
                # Clear existing collection
                try:
                    st.session_state.chroma_client.delete_collection("sds_docs")
                except:
                    pass
                st.session_state.collection = st.session_state.chroma_client.create_collection("sds_docs", embedding_function=multilingual_ef)
                
                # Add documents to ChromaDB
                # We can add in batches if it's very large, but for SDS it should be fine in one go.
                batch_size = 100
                for i in range(0, len(chunks), batch_size):
                    st.session_state.collection.add(
                        documents=chunks[i:i+batch_size],
                        metadatas=metadatas[i:i+batch_size],
                        ids=ids[i:i+batch_size]
                    )
                
                st.session_state.document_processed = True
                st.success(f"Processed! Added {len(chunks)} chunks to the vector database.")
            except Exception as e:
                st.error(f"Error processing chunks: {e}")

# Chat Interface
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask a question about the SDS..."):
    if not st.session_state.document_processed:
        st.warning("Please upload and process a document first.")
    elif not api_key:
        st.error("Please enter your Groq API Key.")
    else:
        # Save and show user message
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
            
        with st.chat_message("assistant"):
            with st.spinner("Searching and Thinking..."):
                try:
                    # 1. Translate Query to Document Language
                    translated_query = prompt
                    try:
                        sample_chunk = st.session_state.collection.get(ids=["chunk_0"])
                        if sample_chunk and sample_chunk['documents']:
                            sample_text = sample_chunk['documents'][0][:500]
                            translation_prompt = f"""
                            You are a translator. I have a document that starts with the following text:
                            "{sample_text}"
                            
                            Translate the following user question into the exact same language as the text above.
                            User question: "{prompt}"
                            
                            Output ONLY the translated question. Do not include any other text, quotes, or explanations.
                            """
                            trans_completion = groq_client.chat.completions.create(
                                messages=[{"role": "user", "content": translation_prompt}],
                                model="llama-3.1-8b-instant",
                            )
                            translated_query = trans_completion.choices[0].message.content.strip().strip('"').strip()
                    except Exception:
                        pass

                    # 2. Query ChromaDB with the TRANSLATED query
                    total_docs = st.session_state.collection.count()
                    n_res = min(total_docs, 4)  # Severely reduced to 4 to stay under 6000 TPM limit
                    
                    results = st.session_state.collection.query(
                        query_texts=[translated_query],
                        n_results=n_res
                    )
                    
                    distances = results['distances'][0]
                    documents = results['documents'][0]
                    metadatas = results['metadatas'][0]
                    
                    # 3. SDS Heuristic: Product Name (Sec 1)
                    valid_chunks = []
                    
                    try:
                        first_chunks = st.session_state.collection.get(
                            ids=["chunk_0", "chunk_1"]
                        )
                        if first_chunks and first_chunks['documents']:
                            for doc, meta in zip(first_chunks['documents'], first_chunks['metadatas']):
                                valid_chunks.append(f"[Page {meta['page']}] {doc}")
                    except Exception:
                        pass

                    # Add the semantically searched chunks (Filtering by Relevance Threshold)
                    RELEVANCE_THRESHOLD = 1.5 # Maximum acceptable L2 distance
                    
                    for doc, meta, dist in zip(documents, metadatas, distances):
                        if dist <= RELEVANCE_THRESHOLD:
                            chunk_text = f"[Page {meta['page']}] {doc}"
                            if chunk_text not in valid_chunks:
                                valid_chunks.append(chunk_text)
                            
                    if not valid_chunks:
                        answer = "I'm sorry, but I couldn't find relevant information in the document to answer your question."
                        st.markdown(answer)
                        st.session_state.chat_history.append({"role": "assistant", "content": answer})
                    else:
                        # Truncate context to exactly 7,000 characters to mathematically guarantee 
                        # we NEVER hit the Groq 6000 TPM limit for dense foreign languages
                        context = "\n\n---\n\n".join(valid_chunks)[:7000]


                        
                        # 3. Ask Groq the question using the context
                        prompt_text = f"""
                        You are a Safety Data Sheet Assistant.
                        
                        STEP 1: Identify the language of the QUESTION below.
                        STEP 2: Read the CONTEXT.
                        STEP 3: Extract the answer from the CONTEXT and TRANSLATE it entirely into the language of the QUESTION.
                        
                        CRITICAL RULES: 
                        - You MUST write your final answer ENTIRELY in the same language as the QUESTION.
                        - DO NOT output any text in the language of the CONTEXT. The user cannot read it.
                        - Always include the [Page X] citation at the end.
                        - If the answer is not in the CONTEXT, reply: "I don't know based on the document."

                        CONTEXT:
                        {context}
                        
                        QUESTION: {prompt}
                        """
                        
                        chat_completion = groq_client.chat.completions.create(
                            messages=[
                                {
                                    "role": "user",
                                    "content": prompt_text,
                                }
                            ],
                            model="llama-3.1-8b-instant",
                        )
                        answer = chat_completion.choices[0].message.content
                        
                        # Show answer
                        st.markdown(answer)
                        with st.expander("🔍 View Retrieved Context (For Debugging)"):
                            st.write("Here is the exact text the AI read to answer your question:")
                            st.text(context)
                        
                        st.session_state.chat_history.append({"role": "assistant", "content": answer})
                        
                except Exception as e:
                    st.error(f"Error generating answer: {e}")
