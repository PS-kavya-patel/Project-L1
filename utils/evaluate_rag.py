import os
import pypdf
import chromadb
from groq import Groq
import pandas as pd

def extract_and_chunk(filepath, chunk_size=1000, overlap=200):
    reader = pypdf.PdfReader(filepath)
    chunks, metadatas, ids = [], [], []
    chunk_id = 0
    for i, page in enumerate(reader.pages):
        page_text = page.extract_text()
        if page_text:
            start = 0
            while start < len(page_text):
                end = start + chunk_size
                chunks.append(page_text[start:end])
                metadatas.append({"page": i + 1})
                ids.append(f"chunk_{chunk_id}")
                chunk_id += 1
                start += chunk_size - overlap
    return chunks, metadatas, ids

def evaluate_pipeline(pdf_path, api_key, test_qa_pairs):
    print(f"Evaluating pipeline on {pdf_path}...")
    
    # 1. Setup ChromaDB
    client = chromadb.Client()
    from chromadb.utils import embedding_functions
    multilingual_ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="paraphrase-multilingual-MiniLM-L12-v2")
    try:
        client.delete_collection("eval_docs")
    except:
        pass
    collection = client.create_collection("eval_docs", embedding_function=multilingual_ef)
    
    chunks, metadatas, ids = extract_and_chunk(pdf_path)
    print(f"Extracted {len(chunks)} chunks.")
    
    batch_size = 100
    for i in range(0, len(chunks), batch_size):
        collection.add(
            documents=chunks[i:i+batch_size],
            metadatas=metadatas[i:i+batch_size],
            ids=ids[i:i+batch_size]
        )
    print("Indexed in ChromaDB.")
    
    # 2. Setup Groq
    groq_client = Groq(api_key=api_key)
    
    results = []
    
    # 3. Evaluate Questions
    for qa in test_qa_pairs:
        question = qa["question"]
        expected = qa["expected_concept"]
        should_reject = qa.get("should_reject", False)
        
        print(f"\nQuestion: {question}")
        
        # Retrieve
        search_res = collection.query(query_texts=[question], n_results=3)
        docs = search_res["documents"][0]
        distances = search_res["distances"][0]
        metas = search_res["metadatas"][0]
        
        threshold = 1.5
        valid_chunks = [f"[Page {m['page']}] {d}" for d, m, dist in zip(docs, metas, distances) if dist < threshold]
        
        retrieval_success = len(valid_chunks) > 0
        
        if not retrieval_success:
            answer = "I'm sorry, but I couldn't find relevant information in the document to answer your question."
            if should_reject:
                response_score = 1 # Correctly rejected
                print("Correctly rejected irrelevant question.")
            else:
                response_score = 0
        else:
            context = "\n\n".join(valid_chunks)
            prompt = f"""
            You are an SDS Assistant. Answer using ONLY the context. Cite page numbers.
            CONTEXT: {context}
            QUESTION: {question}
            """
            completion = groq_client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model="llama-3.1-8b-instant"
            )
            answer = completion.choices[0].message.content
            
            if should_reject:
                response_score = 0 # It should have rejected but didn't
            else:
                # Use LLM as judge
                judge_prompt = f"""
                Evaluate if the generated answer correctly addresses the question based on the expected concept.
                Question: {question}
                Expected Concept: {expected}
                Generated Answer: {answer}
                
                Does the generated answer accurately contain the expected concept?
                Answer only with YES or NO.
                """
                judge_comp = groq_client.chat.completions.create(
                    messages=[{"role": "user", "content": judge_prompt}],
                    model="llama-3.1-8b-instant"
                )
                judge_res = judge_comp.choices[0].message.content.strip().upper()
                response_score = 1 if "YES" in judge_res else 0
            
        print(f"Generated Answer: {answer}")
        
        results.append({
            "Question": question,
            "Retrieval_Success": not should_reject == retrieval_success, # True if it retrieved when it should have, and didn't when it shouldn't
            "Response_Accurate": response_score == 1
        })
        
    df = pd.DataFrame(results)
    print("\n--- Evaluation Summary ---")
    print(df)
    print(f"\nOverall Context Retrieval Accuracy: {df['Retrieval_Success'].mean() * 100:.1f}%")
    print(f"Overall Generation Accuracy: {df['Response_Accurate'].mean() * 100:.1f}%")

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print("Please set GROQ_API_KEY in .env file or environment variables.")
        exit(1)
        
    # Example usage (requires a sample PDF in the directory)
    # You can customize this path and QA pairs for your specific SDS
    test_pdf = "sample_sds.pdf" 
    
    if not os.path.exists(test_pdf):
        print(f"No test PDF found at '{test_pdf}'. Please place an SDS PDF named 'sample_sds.pdf' in the directory to run evaluation.")
    else:
        test_qa = [
            {"question": "What is the product name?", "expected_concept": "The name of the chemical or product.", "should_reject": False},
            {"question": "What are the first aid measures for skin contact?", "expected_concept": "Wash with water/soap.", "should_reject": False},
            {"question": "What is the capital of France?", "expected_concept": "Should reject as it's not in SDS.", "should_reject": True}
        ]
        evaluate_pipeline(test_pdf, api_key, test_qa)
