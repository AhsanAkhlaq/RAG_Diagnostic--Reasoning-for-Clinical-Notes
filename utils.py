import json
import os
from pathlib import Path
from typing import List, Dict, Tuple
import torch
from rank_bm25 import BM25Okapi
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

def load_data(data_path: str) -> Tuple[List[str], List[Dict]]:
    """
    Loads clinical notes from JSON files in the specified directory.
    Parses input sections and combines them into a single text document.
    """
    documents = []
    metadatas = []
    
    # Walk through the directory
    for root, _, files in os.walk(data_path):
        for file in files:
            if file.endswith(".json"):
                file_path = os.path.join(root, file)
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    
                # Combine input sections (input1 to input6)
                full_text = ""
                for i in range(1, 7):
                    key = f"input{i}"
                    if key in data:
                        full_text += f"Section {i}: {data[key]}\n"
                
                if full_text.strip():
                    documents.append(full_text)
                    
                    # Extract diagnosis from path or filename as metadata
                    # Assuming path structure: .../Disease/Type/filename.json
                    path_parts = Path(file_path).parts
                    diagnosis = "Unknown"
                    if len(path_parts) >= 3:
                        diagnosis = f"{path_parts[-3]} - {path_parts[-2]}"
                        
                    metadatas.append({
                        "source": file,
                        "diagnosis": diagnosis,
                        "path": file_path
                    })

                    
    return documents, metadatas

def preprocess_text(text: str) -> List[str]:
    """
    Tokenizes text, removing punctuation and common stop words.
    """
    # Simple list of stop words
    stop_words = set([
        "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by", 
        "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "do", "does", "did",
        "it", "this", "that", "these", "those", "i", "you", "he", "she", "we", "they", "me", "him", "her", "us", "them",
        "what", "which", "who", "whom", "whose", "where", "when", "why", "how",
        "all", "any", "both", "each", "few", "more", "most", "other", "some", "such",
        "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very",
        "can", "will", "just", "should", "now"
    ])
    
    # Remove punctuation and convert to lower case
    text = text.lower()
    for char in '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~':
        text = text.replace(char, ' ')
        
    tokens = text.split()
    return [t for t in tokens if t not in stop_words]

def create_retriever(documents: List[str]):
    """
    Creates a BM25 retriever from the list of documents.
    """
    tokenized_corpus = [preprocess_text(doc) for doc in documents]
    bm25 = BM25Okapi(tokenized_corpus)
    return bm25

def load_llm():
    """
    Loads the TinyLlama model and tokenizer.
    """
    model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    # Use GPU if available, otherwise CPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None,
        low_cpu_mem_usage=True
    )
    
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=512,
        do_sample=True,
        temperature=0.1, # Lower temperature for more factual answers
        top_p=0.9,
        repetition_penalty=1.2
    )
    
    return pipe

def rag_pipeline(query: str, bm25, documents: List[str], metadatas: List[Dict], llm_pipeline, k: int = 3):
    """
    Runs the RAG pipeline: Retrieve -> Generate.
    """
    # 1. Retrieve
    tokenized_query = preprocess_text(query)
    # Get top n document indices
    scores = bm25.get_scores(tokenized_query)
    top_n_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
    
    retrieved_docs = [documents[i] for i in top_n_indices]
    retrieved_metas = [metadatas[i] for i in top_n_indices]
    
    # 2. Prepare Context
    context = ""
    for i, doc in enumerate(retrieved_docs):
        context += f"Document {i+1} (Diagnosis: {retrieved_metas[i]['diagnosis']}):\n{doc[:1500]}...\n\n" # Increased context window slightly
        
    # 3. Generate Prompt
    # TinyLlama chat format
    messages = [
        {
            "role": "system",
            "content": "You are a medical assistant. Answer the user's question strictly based on the provided context. Do not hallucinate or make up information. If the answer is not in the context, simply say 'The provided documents do not contain the answer.'."
        },
        {
            "role": "user",
            "content": f"Context:\n{context}\n\nQuestion: {query}"
        }
    ]
    
    prompt = llm_pipeline.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    # 4. Generate Answer
    outputs = llm_pipeline(prompt)
    generated_text = outputs[0]["generated_text"]
    
    # Extract just the assistant's response
    response = generated_text.split("<|assistant|>")[-1].strip()
    
    return response, retrieved_metas
