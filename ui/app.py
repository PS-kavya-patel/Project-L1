import streamlit as st
import os
import pypdf
import chromadb
from chromadb.utils import embedding_functions
from groq import Groq
import re

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

def extract_text_and_chunk(file, max_chunk_size=1500, overlap=200):
    """Read text from a PDF file, chunk it logically by SDS sections."""
    reader = pypdf.PdfReader(file)
    full_text = ""
    page_map = [] # Track which character index corresponds to which page

    for i, page in enumerate(reader.pages):
        page_text = page.extract_text()
        if page_text:
            page_map.append({"page": i + 1, "start": len(full_text)})
            full_text += page_text + "\n\n"
            
    # Pattern to match SDS sections: 
    # Looks for optional section-like headers in various languages, followed by 1-16 and a separator
    section_pattern = re.compile(r"(?im)^(?:\s*(?:SECTION|ABSCHNITT|SECCIÓN|RUBRIQUE|SEZIONE|HOOFDSTUK|PUNKT)\s+)?([1-9]|1[0-6])[\.\:\-]\s+[A-Z].*$")
    
    matches = list(section_pattern.finditer(full_text))
    
    chunks = []
    metadatas = []
    ids = []
    chunk_id = 0
    
    def get_page_for_idx(idx):
        for p in reversed(page_map):
            if p["start"] <= idx:
                return p["page"]
        return 1
        
    if not matches:
        # Fallback to simple chunking if no sections found
        start = 0
        while start < len(full_text):
            end = start + max_chunk_size
            chunk = full_text[start:end]
            chunks.append(chunk)
            metadatas.append({"page": get_page_for_idx(start), "source": file.name})
            ids.append(f"chunk_{chunk_id}")
            chunk_id += 1
            start += max_chunk_size - overlap
        return chunks, metadatas, ids

    # Chunk by sections
    for i, match in enumerate(matches):
        start_idx = match.start()
        end_idx = matches[i+1].start() if i + 1 < len(matches) else len(full_text)
        
        section_text = full_text[start_idx:end_idx].strip()
        section_header = match.group(0).strip()
        
        # Sub-chunk if section is too large for embeddings
        if len(section_text) > max_chunk_size:
            sub_start = 0
            part = 1
            while sub_start < len(section_text):
                sub_end = sub_start + max_chunk_size
                sub_chunk_text = section_text[sub_start:sub_end]
                
                # Prepend header if it's not the first part to retain semantic link
                if part > 1:
                    sub_chunk_text = f"{section_header} (Continued Part {part})\n...\n{sub_chunk_text}"
                    
                chunks.append(sub_chunk_text)
                metadatas.append({"page": get_page_for_idx(start_idx + sub_start), "source": file.name})
                ids.append(f"chunk_{chunk_id}")
                chunk_id += 1
                
                sub_start += max_chunk_size - overlap
                part += 1
        else:
            chunks.append(section_text)
            metadatas.append({"page": get_page_for_idx(start_idx), "source": file.name})
            ids.append(f"chunk_{chunk_id}")
            chunk_id += 1
            
    # Include anything before the first section as a chunk (intro/title page)
    if matches and matches[0].start() > 0:
        intro_text = full_text[:matches[0].start()].strip()
        if intro_text:
            chunks.insert(0, intro_text)
            metadatas.insert(0, {"page": get_page_for_idx(0), "source": file.name})
            ids = [f"chunk_{i}" for i in range(len(chunks))]

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
                        sample_text = ""
                        for i in range(min(3, st.session_state.collection.count())):
                            chunk = st.session_state.collection.get(ids=[f"chunk_{i}"])
                            if chunk and chunk['documents']:
                                sample_text += chunk['documents'][0][:400] + " "
                        
                        if sample_text.strip():
                            translation_prompt = f"""
                            You are a raw translation engine. I will provide some text from a Safety Data Sheet (SDS).
                            Your task is to identify its language, and translate the USER QUESTION into that EXACT SAME LANGUAGE.
                            
                            SDS TEXT SAMPLE:
                            "{sample_text[:1000]}"
                            
                            USER QUESTION:
                            "{prompt}"
                            
                            CRITICAL RULE: Output ONLY the raw translated question. No intro, no conversational filler, no quotes. Just the translation.
                            """
                            trans_completion = groq_client.chat.completions.create(
                                messages=[{"role": "user", "content": translation_prompt}],
                                model="llama-3.3-70b-versatile",
                            )
                            translated_query = trans_completion.choices[0].message.content.strip().strip('"').strip()
                    except Exception:
                        pass

                    # 2. Query ChromaDB with BOTH original and translated query for robust multilingual search
                    total_docs = st.session_state.collection.count()
                    n_res = min(total_docs, 20)  # Increased to 20 to ensure we capture the correct page
                    
                    query_list = [prompt]
                    if translated_query != prompt and translated_query.strip():
                        query_list.append(translated_query)

                    results = st.session_state.collection.query(
                        query_texts=query_list,
                        n_results=n_res
                    )
                    
                    # Flatten results
                    distances = []
                    documents = []
                    metadatas = []
                    for i in range(len(query_list)):
                        distances.extend(results['distances'][i])
                        documents.extend(results['documents'][i])
                        metadatas.extend(results['metadatas'][i])
                    
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

                    # Add the semantically searched chunks WITH a distance threshold
                    MAX_DISTANCE = 1.2 # Adjust based on embedding model and distance metric
                    for doc, meta, dist in zip(documents, metadatas, distances):
                        if dist > MAX_DISTANCE:
                            continue # Ignore chunks that are too far
                        chunk_text = f"[Page {meta['page']}] {doc}"
                        if chunk_text not in valid_chunks:
                            valid_chunks.append(chunk_text)
                            
                    # 4. Exact Phrase Heuristic (Fallback for specific headings)
                    try:
                        all_docs_results = st.session_state.collection.get()
                        if all_docs_results and all_docs_results['documents']:
                            words = prompt.split()
                            phrases = [" ".join(words[i:i+2]).lower() for i in range(len(words)-1)]
                            if translated_query != prompt:
                                t_words = translated_query.split()
                                phrases.extend([" ".join(t_words[i:i+2]).lower() for i in range(len(t_words)-1)])
                                
                            for doc, meta in zip(all_docs_results['documents'], all_docs_results['metadatas']):
                                doc_lower = doc.lower()
                                match = False
                                for phrase in phrases:
                                    p_words = phrase.split()
                                    if len(p_words) == 2 and len(p_words[0]) > 3 and len(p_words[1]) > 3 and phrase in doc_lower:
                                        match = True
                                        break
                                if match or (translated_query.lower() in doc_lower and len(translated_query) > 5):
                                    chunk_text = f"[Page {meta['page']}] {doc}"
                                    if chunk_text not in valid_chunks:
                                        valid_chunks.insert(0, chunk_text) # Insert at top to ensure it's not truncated
                    except Exception:
                        pass
                            
                    if not valid_chunks:
                        answer = "I don't know based on the document. No relevant information was found."
                        st.markdown(answer)
                        st.session_state.chat_history.append({"role": "assistant", "content": answer})
                    else:
                        # Truncate context to 25,000 characters to ensure we include enough retrieved chunks while staying within limits
                        context = "\n\n---\n\n".join(valid_chunks)[:25000]


                        
                        # 3. Ask Groq the question using the context with proper System Prompting
                        system_prompt_text = f"""
                        You are a strict Safety Data Sheet Assistant.
                        
                        CONTEXT (may be in a different language than the question):
                        {context}
                        
                        CRITICAL INSTRUCTIONS:
                        1. EXTRACT DATA: Read the CONTEXT above and extract the precise information requested in the USER QUESTION. 
                           - If the user asks for an entire SDS section, return EVERY line belonging to that section. Do not summarize. Do not omit subsections. Include 2.1, 2.2, 2.3, etc. Return the text exactly as it appears. Stop only when the next numbered section begins.
                           - If the QUESTION asks for Hazard Statements (H-codes) or Precautionary Statements (P-codes), look specifically in Section 2 (Hazards Identification) of the SDS. Do NOT confuse this with Section 3 (Composition/Ingredients).
                           - Search the CONTEXT thoroughly. Use your multilingual capabilities to match concepts.
                           - If the requested information is NOT in the CONTEXT, you MUST reply exactly: "I don't know based on the document."
                        2. TRANSLATE TO QUESTION LANGUAGE: You must translate the extracted information so that your ENTIRE final response is in the EXACT same language as the USER QUESTION. 
                           - NO EXCEPTIONS. Do not mix languages or leave terms in the original language.
                        3. FORMAT: Present your findings as a complete, exhaustive list. Do not summarize or omit items.
                        4. CITATION: Always include the [Page X] citation at the end.
                        """
                        
                        user_prompt_text = f"{prompt}\n(For reference, the question translated to the document's language is: {translated_query})"
                        
                        chat_completion = groq_client.chat.completions.create(
                            messages=[
                                {
                                    "role": "system",
                                    "content": system_prompt_text,
                                },
                                {
                                    "role": "user",
                                    "content": user_prompt_text,
                                }
                            ],
                            model="llama-3.3-70b-versatile",
                            max_tokens=8000,
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
