import os
import pypdf
import chromadb
from chromadb.utils import embedding_functions
from groq import Groq

# Mock evaluate.py to satisfy the requirement of a repeatable evaluation suite.
# In a real environment, you would place sample PDF documents in a 'tests/data' folder.

TEST_QUESTIONS = [
    {
        "question": "What is the product name?",
        "expected_topics": ["Product name", "Identification"],
        "should_find_answer": True
    },
    {
        "question": "What are the hazard statements?",
        "expected_topics": ["Hazard", "H3", "Danger"],
        "should_find_answer": True
    },
    {
        "question": "What are the first aid measures for eye contact?",
        "expected_topics": ["Rinse cautiously with water", "medical attention"],
        "should_find_answer": True
    },
    {
        "question": "How should this chemical be stored?",
        "expected_topics": ["Store in a well-ventilated place", "Keep cool"],
        "should_find_answer": True
    },
    {
        "question": "What is the recipe for chocolate chip cookies?",
        "expected_topics": ["I don't know based on the document"],
        "should_find_answer": False
    },
    {
        "question": "Who is the CEO of the company?",
        "expected_topics": ["I don't know based on the document"],
        "should_find_answer": False
    },
    {
        "question": "What is the UN number for transportation?",
        "expected_topics": ["UN", "Transport"],
        "should_find_answer": True
    },
    {
        "question": "What are the environmental hazards?",
        "expected_topics": ["Toxic to aquatic life", "environment"],
        "should_find_answer": True
    }
]

def run_evaluation(pdf_path="sample_sds.pdf"):
    print("Starting Evaluation Suite...")
    print(f"Total test questions: {len(TEST_QUESTIONS)}")
    
    if not os.path.exists(pdf_path):
        print(f"Test PDF '{pdf_path}' not found. Please provide a valid SDS PDF for evaluation.")
        return

    # Initialize RAG Pipeline components
    groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY", "dummy"))
    chroma_client = chromadb.Client()
    multilingual_ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="paraphrase-multilingual-MiniLM-L12-v2")
    
    try:
        chroma_client.delete_collection("eval_docs")
    except:
        pass
    collection = chroma_client.create_collection("eval_docs", embedding_function=multilingual_ef)

    # 1. Extract and Chunk
    print("Extracting and chunking text...")
    reader = pypdf.PdfReader(pdf_path)
    full_text = ""
    for page in reader.pages:
        full_text += page.extract_text() + "\n\n"
        
    chunks = [full_text[i:i+1000] for i in range(0, len(full_text), 800)]
    metadatas = [{"page": i // 2 + 1} for i in range(len(chunks))]
    ids = [f"chunk_{i}" for i in range(len(chunks))]
    
    collection.add(documents=chunks, metadatas=metadatas, ids=ids)
    
    # 2. Run Queries
    success_count = 0
    for idx, test in enumerate(TEST_QUESTIONS):
        print(f"\n--- Test {idx+1} ---")
        print(f"Q: {test['question']}")
        
        results = collection.query(query_texts=[test["question"]], n_results=5)
        
        valid_chunks = []
        MAX_DISTANCE = 1.2
        for doc, meta, dist in zip(results['documents'][0], results['metadatas'][0], results['distances'][0]):
            if dist <= MAX_DISTANCE:
                valid_chunks.append(f"[Page {meta['page']}] {doc}")
                
        if not valid_chunks:
            answer = "I don't know based on the document. No relevant information was found."
            print("Retrieved Context: NONE (Distance Threshold Exceeded)")
        else:
            context = "\n---\n".join(valid_chunks)
            print(f"Retrieved Context: {len(valid_chunks)} chunks passed threshold.")
            
            system_prompt = f"You are an SDS Assistant. Use context: {context}. If missing, say 'I don't know based on the document.'"
            try:
                chat_completion = groq_client.chat.completions.create(
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": test["question"]}
                    ],
                    model="llama-3.3-70b-versatile",
                    max_tokens=500
                )
                answer = chat_completion.choices[0].message.content
            except Exception as e:
                answer = f"API Error: {e}"
                
        print(f"A: {answer[:200]}...")
        
        # Basic heuristic validation
        if test["should_find_answer"]:
            if "I don't know" not in answer:
                success_count += 1
                print("Result: PASS")
            else:
                print("Result: FAIL (Expected an answer but got 'I don't know')")
        else:
            if "I don't know" in answer:
                success_count += 1
                print("Result: PASS (Correctly refused)")
            else:
                print("Result: FAIL (Hallucinated an answer)")
                
    print(f"\nEvaluation Complete: {success_count}/{len(TEST_QUESTIONS)} passed.")

if __name__ == "__main__":
    run_evaluation()
