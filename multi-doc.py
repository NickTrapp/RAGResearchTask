from sentence_transformers import SentenceTransformer
import chromadb
import numpy as np
import os
import re
from google import genai
from evaluate import load
import json
import unicodedata
from pdf2image import convert_from_path
import pytesseract
import nltk
from langchain_text_splitters import RecursiveCharacterTextSplitter

# --- 1. Document Ingestion & Preprocessing Module ---

def extract_text_from_pdf(pdf_path):
    """
    Extracts text from each page of a PDF using OCR (pytesseract).
    Returns it as a single string.
    """
    try:
        # pytesseract needs to be installed on your system AND pdf2image needs poppler
        # Ensure 'poppler_path' is set if poppler is not in your system's PATH
        # Example: images = convert_from_path(pdf_path, poppler_path=r'C:\Program Files\poppler\bin')
        images = convert_from_path(pdf_path)
        text = ''
        for page in images:
            text += pytesseract.image_to_string(page) + '\n'
        return text
    except Exception as e:
        print(f"Error during OCR extraction of {pdf_path}: {e}")
        print("Please ensure Tesseract and Poppler are installed and accessible in your PATH.")
        print("For Windows, you might need to set pytesseract.pytesseract.tesseract_cmd manually.")
        # Example for Windows pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
        return ""


def clean_text_from_ocr(ocr_text):
    """
    Cleans raw OCR text output, removing common artifacts
    while preserving readability and word integrity.
    """
    cleaned_text = ocr_text

    # 1. Normalize Unicode and remove problematic invisible characters
    cleaned_text = unicodedata.normalize("NFKC", cleaned_text)
    cleaned_text = cleaned_text.replace('\x00', '')       # Null bytes
    cleaned_text = cleaned_text.replace('\u00A0', ' ')    # Non-breaking space -> standard space
    cleaned_text = cleaned_text.replace('\u200B', '')     # Zero-width space -> remove

    # 2. Remove OCR-specific artifacts:
    # 'e' used as bullet points (often from lists)
    cleaned_text = re.sub(r'^\s*e\s+(\d+\.)', r'\1', cleaned_text, flags=re.MULTILINE) # e 1. -> 1.
    cleaned_text = re.sub(r'^\s*e\s+', '', cleaned_text, flags=re.MULTILINE) # Remove stray 'e ' bullets
    
    # Page break characters (often '' or form feed)
    cleaned_text = cleaned_text.replace('\f', '\n') 

    # Fix hyphenation across lines (e.g., "word- \n word" -> "wordword")
    cleaned_text = re.sub(r'(\w+)-\s*\n\s*(\w+)', r'\1\2', cleaned_text)

    # Remove extra blank lines between words if not intended as paragraph breaks
    cleaned_text = re.sub(r'(\w)\n(\w)', r'\1 \2', cleaned_text) # Single newline between words

    # 3. General whitespace normalization: replace any sequence of whitespace with a single space, then strip.
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()

    # 4. Normalize multiple consecutive blank lines for paragraph breaks
    cleaned_text = re.sub(r'\n\s*\n\s*\n+', '\n\n', cleaned_text)

    return cleaned_text

def chunk_plain_text(text_content, filename, max_chunk_tokens=500, chunk_overlap_tokens=100): # Added filename
    """
    Chunks plain text content by grouping complete sentences.
    Suitable for raw text output from OCR.
    """
    # Step 1: Sentence Segmentation using NLTK
    sentences = nltk.sent_tokenize(text_content) 
    
    chunks_of_sentences = []
    current_chunk_sentences = []
    current_chunk_length = 0 # Using character length as proxy for tokens

    for sentence in sentences:
        sentence_length = len(sentence) 

        # If adding the next sentence makes the chunk too large,
        # and the current chunk is not empty (to avoid single very long sentence forming huge chunk)
        if (current_chunk_length + sentence_length > max_chunk_tokens) and current_chunk_sentences:
            # Add the current group of sentences as a chunk
            chunks_of_sentences.append(" ".join(current_chunk_sentences))
            
            # Reset for a new chunk with overlap
            overlap_sentences = []
            if chunk_overlap_tokens > 0 and current_chunk_sentences:
                temp_len = 0
                for s in reversed(current_chunk_sentences):
                    if temp_len + len(s) <= chunk_overlap_tokens:
                        overlap_sentences.insert(0, s) # Insert at beginning to maintain order
                        temp_len += len(s)
                    else:
                        break # Stop if adding more sentences would exceed overlap
            
            current_chunk_sentences = overlap_sentences + [sentence]
            current_chunk_length = sum(len(s) for s in current_chunk_sentences)
        else:
            current_chunk_sentences.append(sentence)
            current_chunk_length += sentence_length
    
    # Add the very last chunk (if any remaining)
    if current_chunk_sentences:
        chunks_of_sentences.append(" ".join(current_chunk_sentences))

    # Format for ChromaDB with metadata
    processed_chunks = []
    for i, chunk_content in enumerate(chunks_of_sentences):
        processed_chunks.append({
            "content": chunk_content.strip(),
            "original_content": chunk_content,
            "metadata": {
                "chunk_id": i,
                "source": filename, # Store the actual filename for multi-document context
                "type": "sentence_group", # New metadata field
            }
        })
    return processed_chunks

# --- 2. Embedding & Indexing Module ---

def create_vector_db(chunks_with_metadata, model):
    """
    Creates and populates a ChromaDB collection with text chunks and their embeddings.
    """
    client = chromadb.EphemeralClient() # Use PersistentClient if you want to save to disk
    collection_name = "document_qa_collection"

    try:
        collection = client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"} # Use cosine similarity
        )
    except Exception as e:
        print(f"Error creating/getting collection: {e}")
        return None

    print(f"Embedding {len(chunks_with_metadata)} chunks...")
    
    contents_to_embed = [item['content'] for item in chunks_with_metadata]
    metadatas = [item['metadata'] for item in chunks_with_metadata]
    ids = [f"{item['metadata']['source']}_chunk_{item['metadata']['chunk_id']}" for item in chunks_with_metadata]

    embeddings = model.encode(contents_to_embed, show_progress_bar=True)
    embeddings = embeddings.astype('float32') # Ensure float32 for Faiss/ChromaDB compatibility

    collection.add(
        embeddings=embeddings.tolist(),
        documents=contents_to_embed, # Store the chunk content that was embedded
        metadatas=metadatas,
        ids=ids
    )
    print("ChromaDB populated.")
    return collection

# --- 3. Retrieval Module ---

def retrieve_context(query, collection, embedding_model, top_k=5): # Increased default top_k for multi-doc
    """
    Embeds the query and retrieves the top-k most relevant chunks from the ChromaDB.
    Text is assumed to be clean as it was cleaned before embedding.
    """
    query_embedding = embedding_model.encode([query])[0]
    query_embedding = query_embedding.astype('float32')

    results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=top_k,
        include=['documents', 'distances', 'metadatas']
    )

    retrieved_chunks_content = results['documents'][0]
    retrieved_metadatas = results['metadatas'][0]
    retrieved_distances = results['distances'][0]

    print(f"\nRetrieved {len(retrieved_chunks_content)} relevant chunks:")
    
    final_retrieved_chunks_for_llm = []

    for i, chunk_content in enumerate(retrieved_chunks_content):
        metadata = retrieved_metadatas[i]
        distance = retrieved_distances[i]
        source_doc = metadata.get('source', 'Unknown Document') # Get the source filename
        
        print(f"--- Chunk {i+1} (Source: {source_doc}, Distance: {distance:.4f}) ---")
        # Headers metadata will be empty for OCR, but left for consistency
        if 'headers' in metadata and metadata['headers']:
            for header in metadata['headers']:
                print(f"  {header}")
        
        print(f"Content: \n{chunk_content[:500]}...") # Print content up to 500 chars
        
        final_retrieved_chunks_for_llm.append(chunk_content)

    return final_retrieved_chunks_for_llm

# --- 4. Generation Module ---

def generate_answer_with_llm(query, context_chunks):
    """
    Constructs a prompt and calls an LLM to generate an answer.
    You will fill this with your actual LLM API call (e.g., OpenAI, Google Gemini).
    """
    if not context_chunks:
        return "I could not find relevant information in the document to answer your question."

    context = "\n\n".join(context_chunks)

    try:
        client = genai.Client(api_key="AIzaSyDndFoAs7-U_1koyw-nYsnnKM9t9wotGdI")
        prompt = (
              f"You are a helpful assistant specialized in extracting information from documents."
              f"Based ONLY on the following CONTEXT, answer the QUESTION."
              f"If the answer is not explicitly present in the CONTEXT, state clearly that you cannot answer from the provided information.\n\n"
              f"CONTEXT:\n{context}\n\n"
              f"QUESTION: {query}\n\n"
              f"ANSWER:"
        )

        response = client.models.generate_content(
            model="gemini-2.0-flash", contents=prompt
        )
        return response.text
    except Exception as e:
        return f"Error calling LLM: {e}"
    

# --- Evaluation Module ---

def evaluate_system(qa_pairs, db_collection, embedding_model, llm_function, top_k=5): # top_k also here for consistency
    """
    Evaluates the RAG system against a list of question-answer pairs.
    """
    print("\n" + "="*80)
    print("                 STARTING SYSTEM EVALUATION                 ")
    print("="*80 + "\n")

    rouge = load("rouge")
    results_log = []

    for i, qa_pair in enumerate(qa_pairs):
        question = qa_pair['question']
        ground_truth_answer = qa_pair['answer']
        expected_source_document = qa_pair.get('source_document', 'N/A') # Get source if available

        print(f"\n--- Evaluation Case {i+1}/{len(qa_pairs)} ---")
        print(f"Question: {question}")
        print(f"Ground Truth: {ground_truth_answer}")
        print(f"Expected Source: {expected_source_document}") 

        # 1. Retrieval
        retrieved_context_chunks = retrieve_context(question, db_collection, embedding_model, top_k=top_k)

        # 2. Generation
        system_answer = llm_function(question, retrieved_context_chunks)
        print(f"\nSystem Answer: {system_answer}")

        # 3. Evaluation (ROUGE)
        rouge_results = rouge.compute(
            predictions=[system_answer],
            references=[ground_truth_answer]
        )
        print(f"ROUGE Scores: {rouge_results}")

        results_log.append({
            "question": question,
            "ground_truth": ground_truth_answer,
            "expected_source": expected_source_document, # Log expected source
            "retrieved_chunks": retrieved_context_chunks,
            "system_answer": system_answer,
            "rouge_scores": rouge_results
        })
        print("-" * 50)

    print("\n" + "="*80)
    print("                 EVALUATION COMPLETE                  ")
    print("="*80 + "\n")

    with open("evaluation_results_multi_doc.json", "w", encoding="utf-8") as f:
        json.dump(results_log, f, indent=4, ensure_ascii=False)
    print("Evaluation results saved to evaluation_results_multi_doc.json")

    return results_log

# --- Sample QA Dataset for Multi-Document Testing ---
# Place the following PDFs in a folder named 'sample_data' in the same directory as this script:
# 1. 2404.07221v2.pdf (Improving Retrieval for RAG)
# 2. 11316-Article Text-14844-1-2-20201228.pdf (A Unified Model for DBQA)
# 3. SHTI-316-SHTI240567.pdf (Optimizing Data Extraction: German Medical Docs)

sample_qa_dataset = [
    {
        "question": "What are two primary techniques of knowledge injection mentioned in the paper?",
        "answer": "The two primary techniques of knowledge injection are additional training of the model or fine-tuning, or in-context learning, the most popular version of which is Retrieval Augmented Generation (RAG).",
        "expected_source_document": "2404.07221v2.pdf"
    },
    {
        "question": "What is a key disadvantage of LLMs that RAG aims to address?",
        "answer": "A key disadvantage is the tendency for LLMs to hallucinate information and its lack of knowledge in domain specific areas.",
        "expected_source_document": "2404.07221v2.pdf"
    },
    {
        "question": "What evaluation metric is defined by the RAGAS framework to assess retrieved chunk quality?",
        "answer": "Context Relevance is defined by the RAGAS framework to assess retrieved chunk quality.",
        "expected_source_document": "2404.07221v2.pdf"
    },
    {
        "question": "What is the main goal of re-ranking algorithms?",
        "answer": "Re-ranking algorithms is a method to prioritize the relevance over the similarity of the chunks.",
        "expected_source_document": "2404.07221v2.pdf"
    },
    {
        "question": "Which benchmark was used for question and answering models in this paper's results?",
        "answer": "The FinanceBench benchmark was used for question and answering models.",
        "expected_source_document": "2404.07221v2.pdf"
    },
    {
        "question": "What are the three major encoding layers of the unified model?",
        "answer": "The unified model contains three major encoding layers: the basic encoder, combined encoder and hierarchical encoder.",
        "expected_source_document": "11316-Article Text-14844-1-2-20201228.pdf"
    },
    {
        "question": "On which two datasets were extensive experiments conducted for the unified model?",
        "answer": "Extensive experiments were conducted on both the English WikiQA dataset and the Chinese dataset (NLPCC2016).",
        "expected_source_document": "11316-Article Text-14844-1-2-20201228.pdf"
    },
    {
        "question": "What does DBQA stand for?",
        "answer": "DBQA stands for Document-based Question Answering.",
        "expected_source_document": "11316-Article Text-14844-1-2-20201228.pdf"
    },
    {
        "question": "Who is the corresponding author of this paper?",
        "answer": "The corresponding author of this paper is Yunfang Wu.",
        "expected_source_document": "11316-Article Text-14844-1-2-20201228.pdf"
    },
    {
        "question": "What are the three steps of the human-like reading strategy?",
        "answer": "The detailed reading strategy is as follows: 1. Go over the document quickly to get a general understanding. 2. Read the question carefully equipped with the general understanding. 3. Go back to the document with the prior knowledge of question and get the right answer.",
        "expected_source_document": "11316-Article Text-14844-1-2-20201228.pdf"
    },
    {
        "question": "What is the primary challenge in medical informatics discussed in the SHTI240567 paper?",
        "answer": "In medical informatics, processing unstructured text data while ensuring data protection and confidentiality is a major challenge.",
        "expected_source_document": "SHTI-316-SHTI240567.pdf"
    },
    {
        "question": "What is the accuracy of data extraction achieved by the pipeline in the SHTI240567 study?",
        "answer": "The pipeline demonstrated an accuracy of up to 90% in data extraction.",
        "expected_source_document": "SHTI-316-SHTI240567.pdf"
    },
    {
        "question": "What are the key stages of the data extraction pipeline described in the SHTI240567 paper?",
        "answer": "The pipeline works in stages: 1) it inputs unstructured reports, 2) translates them from German to English using an OSS translation model, 3) uses RAG to identify and retrieve information, and 4) then converts these snippets into structured data for downstream use.",
        "expected_source_document": "SHTI-316-SHTI240567.pdf"
    },
    {
        "question": "What languages are involved in the data translation process of the SHTI240567 pipeline?",
        "answer": "The pipeline translates documents from German to English.",
        "expected_source_document": "SHTI-316-SHTI240567.pdf"
    },
    # Negative example (unanswerable from any of the provided documents)
    {
        "question": "Who is the current President of the United States?",
        "answer": "I cannot answer from the provided information.",
        "expected_source_document": "N/A"
    }
]

if __name__ == "__main__":
    print("Loading SentenceTransformer model...")
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

    # --- Setup for Multi-Document ---
    documents_directory = "sample_data" # Create this folder and place your PDFs inside!
    
    if not os.path.exists(documents_directory):
        print(f"Error: Directory '{documents_directory}' not found. Please create it and place your PDFs inside.")
        exit()

    all_processed_chunks_for_db = []
    
    # Loop through each PDF file in the directory
    for filename in os.listdir(documents_directory):
        if filename.lower().endswith(".pdf"):
            pdf_path = os.path.join(documents_directory, filename)
            print(f"\n--- Processing Document: {filename} ---")

            # 1. Document Ingestion (OCR extraction)
            raw_ocr_text = extract_text_from_pdf(pdf_path)
            if not raw_ocr_text:
                print(f"Failed to extract text via OCR from {filename}. Skipping.")
                continue

            # 2. Cleaning OCR Text
            cleaned_full_text = clean_text_from_ocr(raw_ocr_text)
            
            # Optional: Save cleaned text of each document for inspection
            # with open(f"cleaned_{filename.replace('.pdf', '.txt')}", "w", encoding="utf-8") as f:
            #     f.write(cleaned_full_text)

            # 3. Chunking (using the plain text chunker)
            current_doc_chunks = chunk_plain_text(
                cleaned_full_text,
                filename, # Pass the filename to chunker
                max_chunk_tokens=500,
                chunk_overlap_tokens=100
            )
            
            all_processed_chunks_for_db.extend(current_doc_chunks)

    print(f"\nTotal documents processed. Accumulated {len(all_processed_chunks_for_db)} chunks from all PDFs.")

    # 4. Embedding & Indexing (using the accumulated chunks)
    db_collection = create_vector_db(all_processed_chunks_for_db, embedding_model)
    if db_collection is None:
        print("Failed to create vector database. Exiting.")
        exit()

    # 5. Evaluation
    evaluation_results = evaluate_system(
        qa_pairs=sample_qa_dataset,
        db_collection=db_collection,
        embedding_model=embedding_model,
        llm_function=generate_answer_with_llm,
        top_k=5 # Increased top_k for multi-document retrieval
    )