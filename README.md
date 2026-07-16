# Simple SDS AI (Semantic RAG)

A lightweight Retrieval-Augmented Generation (RAG) application built with Python, Streamlit, and ChromaDB. This app allows you to upload Safety Data Sheet (SDS) PDFs, automatically processes them using ChromaDB for semantic vector search, and answers your questions using the Groq API.

![App Screenshot 1](./App%20Screenshot%201.png)
![App Screenshot 2](./App%20Screenshot%202.png)

## Features
- **Semantic Vector Search:** Uses `chromadb` for storing embeddings and performing semantic search on your documents.
- **Source Citations:** Automatically cites page numbers from the source document to ground its answers.
- **Relevance Threshold:** Filters out unrelated questions to prevent hallucinations and indicate when the document lacks context.
- **Fast Generation:** Uses Groq's API (`llama-3.1-8b-instant`) to instantly generate answers based strictly on your document.

## How It Works
1. **Upload:** Upload an SDS PDF file.
2. **Chunk & Index:** The text is extracted, chunked, and embedded into a local vector database using ChromaDB.
3. **Search:** Your question is compared against the document embeddings to find the most relevant sections using distance metrics.
4. **Answer:** The top relevant chunks (if they pass the relevance threshold) are sent to the Groq LLM to generate an accurate answer based solely on the document.

### Architecture Flowchart

```text
 ┌──────────────────┐       ┌──────────────────┐       ┌───────────────────────┐
 │                  │       │                  │       │                       │
 │  1. Upload PDF   ├──────►│ 2. Extract Text  ├──────►│ 3. Chunk & Embed Text │
 │                  │       │                  │       │                       │
 └──────────────────┘       └──────────────────┘       └──────────┬────────────┘
                                                                  │
                                                                  ▼
 ┌──────────────────┐       ┌──────────────────┐       ┌──────────┴────────────┐
 │                  │       │                  │       │                       │
 │ 6. Groq LLM      │◄──────┤ 5. Native Vector │◄──────┤ 4. Local ChromaDB     │
 │    Generation    │       │    Search        │       │    (Vector Store)     │
 └────────┬─────────┘       └─────────▲────────┘       └───────────────────────┘
          │                           │
          │                           │ (Translates Query)
          ▼                           │
 ┌────────┴─────────┐       ┌─────────┴────────┐
 │                  │       │                  │
 │ 7. Final Answer  │       │ User Asks a      │
 │    & Citations   │       │ Question         │
 └──────────────────┘       └──────────────────┘
```


## Cloud-Data Privacy Notice
- **Local Indexing:** The extraction, chunking, and embedding processes, as well as the vector database (ChromaDB), run locally on your machine.
- **Cloud Translation:** To ensure accurate multilingual search, your search query and the first paragraph of your document (up to 500 characters) are sent to the Groq API to detect the document's language and translate your query. 
- **Cloud Generation:** To generate the final answer, the retrieved snippets of your document (the relevant context) and your translated query are sent to the Groq API again. The processing is not strictly local, as this data is shared with Groq (a cloud service) for translation and generation.

## Setup Instructions
1. Clone the repository.
2. Create a virtual environment: `python -m venv venv`
3. Activate the virtual environment:
   - Windows: `.\venv\Scripts\activate`
   - Mac/Linux: `source venv/bin/activate`
4. Install dependencies: `pip install -r requirements.txt`
5. Run the app: `streamlit run app.py`

## Requirements
To run this project, you will need a [Groq API Key](https://console.groq.com/keys).
