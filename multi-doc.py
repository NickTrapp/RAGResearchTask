from sentence_transformers import SentenceTransformer
import chromadb
import os
import re
from google import genai
from evaluate import load
import json
import unicodedata
from pdf2image import convert_from_path
import pytesseract
import nltk

# --- Document Ingestion & Preprocessing Module ---

def extract_text_from_pdf(pdf_path):

    images = convert_from_path(pdf_path)
    text = ''
    for page in images:
        text += pytesseract.image_to_string(page) + '\n'
    return text


def clean_text_from_ocr(ocr_text):
  
    cleaned_text = ocr_text

   
    cleaned_text = unicodedata.normalize("NFKC", cleaned_text)
    cleaned_text = cleaned_text.replace('\x00', '')     
    cleaned_text = cleaned_text.replace('\u00A0', ' ')
    cleaned_text = cleaned_text.replace('\u200B', '')    


    cleaned_text = re.sub(r'^\s*e\s+(\d+\.)', r'\1', cleaned_text, flags=re.MULTILINE)
    cleaned_text = re.sub(r'^\s*e\s+', '', cleaned_text, flags=re.MULTILINE)
    
    cleaned_text = cleaned_text.replace('\f', '\n') 

    cleaned_text = re.sub(r'(\w+)-\s*\n\s*(\w+)', r'\1\2', cleaned_text)

    cleaned_text = re.sub(r'(\w)\n(\w)', r'\1 \2', cleaned_text)

    cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()

    cleaned_text = re.sub(r'\n\s*\n\s*\n+', '\n\n', cleaned_text)

    return cleaned_text

def chunk_plain_text(text_content, filename, max_chunk_tokens=500, chunk_overlap_tokens=100):

    sentences = nltk.sent_tokenize(text_content) 
    
    chunks_of_sentences = []
    current_chunk_sentences = []
    current_chunk_length = 0

    for sentence in sentences:
        sentence_length = len(sentence) 

        if (current_chunk_length + sentence_length > max_chunk_tokens) and current_chunk_sentences:
            chunks_of_sentences.append(" ".join(current_chunk_sentences))
            
            overlap_sentences = []
            if chunk_overlap_tokens > 0 and current_chunk_sentences:
                temp_len = 0
                for s in reversed(current_chunk_sentences):
                    if temp_len + len(s) <= chunk_overlap_tokens:
                        overlap_sentences.insert(0, s)
                        temp_len += len(s)
                    else:
                        break
            
            current_chunk_sentences = overlap_sentences + [sentence]
            current_chunk_length = sum(len(s) for s in current_chunk_sentences)
        else:
            current_chunk_sentences.append(sentence)
            current_chunk_length += sentence_length
    
    if current_chunk_sentences:
        chunks_of_sentences.append(" ".join(current_chunk_sentences))

    processed_chunks = []
    for i, chunk_content in enumerate(chunks_of_sentences):
        processed_chunks.append({
            "content": chunk_content.strip(),
            "original_content": chunk_content,
            "metadata": {
                "chunk_id": i,
                "source": filename,
                "type": "sentence_group",
            }
        })
    return processed_chunks


# --- Embedding & Indexing Module ---

def create_vector_db(chunks_with_metadata, model):

    client = chromadb.EphemeralClient()
    collection_name = "document_qa_collection"

    
    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"}
    )

    print(f"Embedding {len(chunks_with_metadata)} chunks...")
    
    contents_to_embed = [item['content'] for item in chunks_with_metadata]
    metadatas = [item['metadata'] for item in chunks_with_metadata]
    ids = [f"{item['metadata']['source']}_chunk_{item['metadata']['chunk_id']}" for item in chunks_with_metadata]

    embeddings = model.encode(contents_to_embed, show_progress_bar=True)
    embeddings = embeddings.astype('float32')

    collection.add(
        embeddings=embeddings.tolist(),
        documents=contents_to_embed,
        metadatas=metadatas,
        ids=ids
    )
    print("ChromaDB populated.")
    return collection


# --- Retrieval Module ---

def retrieve_context(query, collection, embedding_model, top_k=5):

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
        source_doc = metadata.get('source', 'Unknown Document')
        
        print(f"--- Chunk {i+1} (Source: {source_doc}, Distance: {distance:.4f}) ---")

        if 'headers' in metadata and metadata['headers']:
            for header in metadata['headers']:
                print(f"  {header}")
        
        print(f"Content: \n{chunk_content[:500]}...")
        
        final_retrieved_chunks_for_llm.append(chunk_content)

    return final_retrieved_chunks_for_llm


# --- Generation Module ---

def generate_answer_with_llm(query, context_chunks):

    context = "\n\n".join(context_chunks)

    
    client = genai.Client(api_key="AIzaSyDndFoAs7-U_1koyw-nYsnnKM9t9wotGdI")
    prompt = (
        f"You are an expert research assistant specialized in extracting precise information from provided documents. "
        f"Your primary goal is to answer the user's QUESTION based **STRICTLY AND ONLY** on the CONTEXT provided. "
        f"Adhere to the context faithfully; **DO NOT** use any outside knowledge or prior training data. "
        f"If the answer is not explicitly present or cannot be directly inferred from the CONTEXT, you **MUST** state 'I cannot answer from the provided information.'\n\n"

        f"CONTEXT:\n{context}\n\n"
        f"QUESTION: {query}\n\n"
        
        f"GUIDELINES FOR ANSWERING:\n"
        f"- Be concise but comprehensive. Provide all relevant details from the context."
        f"- If the question asks for a list, steps, or stages, provide them in a clear, numbered list format."
        f"- Avoid conversational filler or apologies; go straight to the answer."
        f"- If the context contains multiple possible answers or interpretations for the question (e.g., different types of disadvantages), prioritize the most direct and specific one mentioned in the context, or list all relevant ones if that's what the question implies."
        f"- If numerical values are requested, provide the exact numbers from the text."
        f"\nANSWER:"

    )

    response = client.models.generate_content(
        model="gemini-2.0-flash", contents=prompt
    )
    return response.text

    


def evaluate_system(qa_pairs, db_collection, embedding_model, llm_function, top_k=5):

    print("\n" + "="*80)
    print("                 STARTING SYSTEM EVALUATION                 ")
    print("="*80 + "\n")

    rouge = load("rouge")
    results_log = []

    for i, qa_pair in enumerate(qa_pairs):
        question = qa_pair['question']
        ground_truth_answer = qa_pair['answer']
        expected_source_document = qa_pair.get('expected_source_document', 'N/A')

        print(f"\n--- Evaluation Case {i+1}/{len(qa_pairs)} ---")
        print(f"Question: {question}")
        print(f"Ground Truth: {ground_truth_answer}")
        print(f"Expected Source: {expected_source_document}") 

        retrieved_context_chunks = retrieve_context(question, db_collection, embedding_model, top_k=top_k)

        system_answer = llm_function(question, retrieved_context_chunks)
        print(f"\nSystem Answer: {system_answer}")

        rouge_results = rouge.compute(
            predictions=[system_answer],
            references=[ground_truth_answer]
        )
        print(f"ROUGE Scores: {rouge_results}")

        results_log.append({
            "question": question,
            "ground_truth": ground_truth_answer,
            "expected_source": expected_source_document,
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



sample_qa_dataset = [
    {
        "question": "What are two primary techniques of knowledge injection in Large Language Models?",
        "answer": "The two primary techniques of knowledge injection are additional training of the model or fine-tuning, or in-context learning, the most popular version of which is Retrieval Augmented Generation (RAG).",
        "expected_source_document": "2404.07221v2.pdf"
    },
    {
        "question": "What is a key disadvantage of Large Language Models that Retrieval Augmented Generation (RAG) aims to address?",
        "answer": "A key disadvantage is the tendency for LLMs to hallucinate information and its lack of knowledge in domain specific areas.",
        "expected_source_document": "2404.07221v2.pdf"
    },
    {
        "question": "What evaluation metric is recommended by the RAGAS framework for assessing retrieved chunk quality in unstructured data?",
        "answer": "Context Relevance is defined by the RAGAS framework to assess retrieved chunk quality.",
        "expected_source_document": "2404.07221v2.pdf"
    },
    {
        "question": "What is the main purpose of re-ranking algorithms in RAG pipelines?",
        "answer": "Re-ranking algorithms is a method to prioritize the relevance over the similarity of the chunks.",
        "expected_source_document": "2404.07221v2.pdf"
    },
    {
        "question": "What kind of benchmark dataset was used to assess the performance of a RAG model in the context of financial documents?",
        "answer": "The FinanceBench benchmark was used for question and answering models.",
        "expected_source_document": "2404.07221v2.pdf"
    },
    {
        "question": "Describe the human-like reading strategy proposed for Document-based Question Answering (DBQA).",
        "answer": "The detailed reading strategy is as follows: 1. Go over the document quickly to get a general understanding. 2. Read the question carefully equipped with the general understanding. 3. Go back to the document with the prior knowledge of question and get the right answer.",
        "expected_source_document": "11316-Article Text-14844-1-2-20201228.pdf"
    },
    {
        "question": "What are the major encoding layers that compose a unified model for DBQA?",
        "answer": "The unified model contains three major encoding layers: the basic encoder, combined encoder and hierarchical encoder.",
        "expected_source_document": "11316-Article Text-14844-1-2-20201228.pdf"
    },
    {
        "question": "Which two specific datasets were used for extensive experiments to evaluate a unified model for DBQA?",
        "answer": "Extensive experiments were conducted on both the English WikiQA dataset and the Chinese dataset (NLPCC2016).",
        "expected_source_document": "11316-Article Text-14844-1-2-20201228.pdf"
    },
    {
        "question": "What does DBQA stand for in the context of natural language processing?",
        "answer": "DBQA stands for Document-based Question Answering.",
        "expected_source_document": "11316-Article Text-14844-1-2-20201228.pdf"
    },
    {
        "question": "What is a key contribution of the unified model based on human-like reading strategy?",
        "answer": "The unified model proposes a human-like reading strategy for DBQA task which is similar to the logic of students when they do the test of reading comprehension, and it combines general understanding of both document and question.",
        "expected_source_document": "11316-Article Text-14844-1-2-20201228.pdf"
    },
    {
        "question": "What is the primary challenge in medical informatics that involves processing unstructured text data?",
        "answer": "In medical informatics, processing unstructured text data while ensuring data protection and confidentiality is a major challenge.",
        "expected_source_document": "SHTI-316-SHTI240567.pdf"
    },
    {
        "question": "What accuracy was achieved by the automated pipeline in data extraction from medical reports in the SHTI240567 study?",
        "answer": "The pipeline demonstrated an accuracy of up to 90% in data extraction.",
        "expected_source_document": "SHTI-316-SHTI240567.pdf"
    },
    {
        "question": "Describe the key stages of the data extraction pipeline for German medical documents.",
        "answer": "The pipeline works in stages: 1) it inputs unstructured reports, 2) translates them from German to English using an OSS translation model, 3) uses RAG to identify and retrieve information, and 4) then converts these snippets into structured data for downstream use.",
        "expected_source_document": "SHTI-316-SHTI240567.pdf"
    },
    {
        "question": "What are the source and target languages for data translation in the medical data extraction pipeline?",
        "answer": "The pipeline translates documents from German to English.",
        "expected_source_document": "SHTI-316-SHTI240567.pdf"
    },
    {
        "question": "Who is the current President of the United States?",
        "answer": "I cannot answer from the provided information.",
        "expected_source_document": "N/A"
    }
]

if __name__ == "__main__":
    print("Loading SentenceTransformer model...")
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

    documents_directory = "sample_data"

    all_processed_chunks_for_db = []
    
    for filename in os.listdir(documents_directory):
        if filename.lower().endswith(".pdf"):
            pdf_path = os.path.join(documents_directory, filename)
            print(f"\n--- Processing Document: {filename} ---")

            raw_ocr_text = extract_text_from_pdf(pdf_path)
            if not raw_ocr_text:
                print(f"Failed to extract text via OCR from {filename}. Skipping.")
                continue

            cleaned_full_text = clean_text_from_ocr(raw_ocr_text)
            

            current_doc_chunks = chunk_plain_text(
                cleaned_full_text,
                filename,
                max_chunk_tokens=500,
                chunk_overlap_tokens=100
            )
            
            all_processed_chunks_for_db.extend(current_doc_chunks)

    print(f"\nTotal documents processed. Accumulated {len(all_processed_chunks_for_db)} chunks from all PDFs.")

    db_collection = create_vector_db(all_processed_chunks_for_db, embedding_model)

    evaluation_results = evaluate_system(
        qa_pairs=sample_qa_dataset,
        db_collection=db_collection,
        embedding_model=embedding_model,
        llm_function=generate_answer_with_llm,
        top_k=3
    )