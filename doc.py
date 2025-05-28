#import pymupdf
import pymupdf4llm
from sentence_transformers import SentenceTransformer
import chromadb
import numpy as np
import os
import re
from google import genai
from evaluate import load
import json # For saving/loading results if needed
import unicodedata
from pdf2image import convert_from_path
import pytesseract
#from PIL import Image

# Corrected imports for two-stage splitting
from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_text_splitters import RecursiveCharacterTextSplitter # You were right to keep this!

# --- 1. Document Ingestion & Preprocessing Module ---

# def extract_markdown_from_pdf(pdf_path):
#     """
#     Extracts text from a PDF document in Markdown format using pymupdf4llm.
#     Handles basic text and structural elements like headings, lists, and tables.
#     """
#     try:
#         md_text = pymupdf4llm.to_markdown(pdf_path)
#         return md_text
#     except Exception as e:
#         print(f"Error extracting markdown from PDF: {e}")
#         return None

def extract_text_from_pdf(pdf_path):
    """
    Extracts text from each page of a PDF and returns it as a single string.
    """
    images = convert_from_path(pdf_path)
    text = ''
    for page in images:
        text += pytesseract.image_to_string(page) + '\n'
    return text


def markdown_aware_chunk_text(markdown_text, max_chunk_tokens=300, chunk_overlap_tokens=50):
    """
    Chunks markdown text using a two-stage approach:
    1. Splits by Markdown headers (MarkdownHeaderTextSplitter).
    2. Further splits overly large header-based chunks by characters (RecursiveCharacterTextSplitter).
    """
    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
        ("####", "Header 4"),
    ]

    # Stage 1: Split by Markdown headers
    markdown_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on
    )
    # This will return a list of Documents, where each Document's page_content
    # is a section of text corresponding to a header level.
    # The metadata of these Documents will contain the headers that define them.
    header_split_documents = markdown_splitter.split_text(markdown_text)

    # Stage 2: Further split these header-defined documents into smaller chunks
    # using a character-based splitter, respecting max_chunk_tokens and overlap.
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=max_chunk_tokens,
        chunk_overlap=chunk_overlap_tokens,
        length_function=len # Use character count. For LLM token count, use appropriate tokenizer.
    )

    # The split_documents method takes a list of Document objects and splits them
    # while preserving their metadata.
    final_chunks_docs = text_splitter.split_documents(header_split_documents)

    processed_chunks = []
    for i, chunk_doc in enumerate(final_chunks_docs):
        chunk_content = chunk_doc.page_content

        # Extract meaningful headers from metadata and join them into a single string
        # The metadata now comes from the header_split_documents, propagated by text_splitter
        header_items = []
        for key, value in chunk_doc.metadata.items():
            if key.startswith("Header") and value is not None and value.strip() != "":
                header_items.append(f"{key}: {value}")

        header_context_str = "; ".join(header_items)

        # Prepend headers to the chunk content for better context in retrieval and LLM.
        # This is your specific logic to enrich the chunk.
        if header_context_str:
            full_chunk_for_embedding = f"{header_context_str}\n\n{chunk_content}"
        else:
            full_chunk_for_embedding = chunk_content

        processed_chunks.append({
            "content": full_chunk_for_embedding, # This is what gets embedded
            "original_content": chunk_content,   # Keep original clean content if needed
            "metadata": {
                "chunk_id": i,
                "source": "pdf",
                "headers": header_context_str, # Store the joined string of headers
                # Add any other useful metadata that might be propagated (e.g., 'source')
                # 'source': chunk_doc.metadata.get('source', 'pdf') # Example for propagated metadata
            }
        })

    return processed_chunks

def clean_markdown_text(markdown_text):



    # 1. Remove non-standard Unicode characters (like those squares/question marks)

    # This regex matches characters that are not basic Latin letters, numbers,

    # common punctuation, or standard whitespace.

    # Be careful not to remove valid non-ASCII characters if they are relevant (e.g., in names).

    # For this paper, it seems safe to remove most non-basic-ascii.

    # Pattern to keep: printable ASCII characters (0x20-0x7E), common whitespace (\t\n\r),

    # and a few common Unicode ranges if needed (e.g., for authors' names with accents).

    # For simplicity, let's target non-printable and known problematic characters first.


    # Remove null bytes, typically seen from some PDF extractions

    cleaned_text = markdown_text.replace('\x00', '')


    # Replace common problematic non-breaking spaces or zero-width spaces

    cleaned_text = cleaned_text.replace('\u00A0', ' ') # Non-breaking space

    cleaned_text = cleaned_text.replace('\u200B', '') # Zero-width space


    # Replace characters that are specifically causing the '?' or 'square' blocks

    # This might require some trial and error if you encounter more types.

    # A common approach is to normalize and then filter.

    cleaned_text = unicodedata.normalize("NFKC", cleaned_text) # Normalize unicode


    # Aggressive removal of non-ASCII characters and potentially problematic control characters

    # Only keep printable ASCII, common whitespace, and some basic Markdown characters

    # You might need to adjust this regex based on what you observe.

    # This pattern keeps:

    # - a-zA-Z0-9 : basic alphanumeric

    # - \s : any whitespace (space, tab, newline, etc.)

    # - \.,:;!?()\[\]{}'" : common punctuation

    # - \-\*#`_ : common markdown symbols

    # - & : for entities like &amp;

    cleaned_text = re.sub(r'[^\x20-\x7E\s.,:;!?()\[\]{}"\'\-*#`_&]', '', cleaned_text)



    # 2. Clean up repetitive blank lines or malformed table structures (e.g., "|<br>|Col2|")

    # Remove lines that are just Markdown table separators or have only spaces/tabs/pipes

    cleaned_text = re.sub(r'^\s*\|[-—]+\|?\s*$', '', cleaned_text, flags=re.MULTILINE) # Remove lines like |---|---|

    cleaned_text = re.sub(r'^\s*\|\s*$', '', cleaned_text, flags=re.MULTILINE) # Remove lines with just a pipe

    cleaned_text = re.sub(r'^\s*<br>\s*$', '', cleaned_text, flags=re.MULTILINE) # Remove empty <br> lines


    # Remove multiple consecutive blank lines, replacing with at most two newlines for paragraph breaks

    cleaned_text = re.sub(r'\n\s*\n\s*\n+', '\n\n', cleaned_text)



    # Remove very short lines that might be remnants of bad parsing (e.g., single characters from equations)

    # This is a heuristic and can be risky if valid short lines exist.

    # Consider doing this after initial chunking if it's too aggressive here.

    # cleaned_text = re.sub(r'^\s*([a-zA-Z0-9])\s*$', '', cleaned_text, flags=re.MULTILINE) # Removes single char lines


    # Replace multiple spaces with a single space (general cleanup)

    cleaned_text = re.sub(r'[ \t]+', ' ', cleaned_text).strip()



    return cleaned_text



# --- 2. Embedding & Indexing Module ---

def create_vector_db(chunks_with_metadata, model):
    """
    Creates and populates a ChromaDB collection with text chunks and their embeddings.
    """
    client = chromadb.EphemeralClient() # Or chromadb.PersistentClient(path="./my_chroma_db")
    collection_name = "document_qa_collection"

    try:
        collection = client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"} # Use cosine similarity
        )
    except Exception as e:
        print(f"Error creating/getting collection: {e}")
        return None

    # Clear existing data if it's an ephemeral client, or if you want to re-add
    # if not isinstance(client, chromadb.PersistentClient): # Don't clear persistent data automatically
    #     collection.delete(ids=collection.get()['ids']) # Clear all existing data if any

    print(f"Embedding {len(chunks_with_metadata)} chunks...")

    # Extract content and metadata separately for embedding
    contents_to_embed = [item['content'] for item in chunks_with_metadata]
    metadatas = [item['metadata'] for item in chunks_with_metadata]
    ids = [f"chunk_{item['metadata']['chunk_id']}" for item in chunks_with_metadata]

    embeddings = model.encode(contents_to_embed, show_progress_bar=True)
    embeddings = embeddings.astype('float32') # Ensure float32 for Faiss/ChromaDB compatibility

    # Add to ChromaDB
    collection.add(
        embeddings=embeddings.tolist(),
        documents=contents_to_embed, # Store the chunk content that was embedded
        metadatas=metadatas,
        ids=ids
    )
    print("ChromaDB populated.")
    return collection

# --- 3. Retrieval Module ---

def retrieve_context(query, collection, embedding_model, top_k=3):
    """
    Embeds the query and retrieves the top-k most relevant chunks from the ChromaDB.
    """
    query_embedding = embedding_model.encode([query])[0]
    query_embedding = query_embedding.astype('float32')

    results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=top_k,
        include=['documents', 'distances', 'metadatas']
    )



    retrieved_chunks_content = results['documents'][0]

    print(f"\nRetrieved {len(retrieved_chunks_content)} relevant chunks:")

    final_retrieved_chunks_for_llm = [] # Use this to store the clean chunks for the LLM

    for i, raw_chunk_from_db in enumerate(retrieved_chunks_content):


        # The primary cleaning should happen *before* embedding.
        # Here, we can do minimal final normalization if necessary.
        # If pre-cleaning is thorough, most of this might not be needed.
        cleaned_chunk_content = raw_chunk_from_db

        # 1. Normalize Unicode (NFKC for compatibility, NFKD for decomposition)
        # Often useful for characters that look similar but are encoded differently.
        cleaned_chunk_content = unicodedata.normalize("NFKC", cleaned_chunk_content)

        # Optional: Remove any remaining control characters that might have slipped through or were generated.
        # This is a lighter version than the one in clean_markdown_text.
        # cleaned_chunk_content = re.sub(r'[\x00-\x1F\x7F-\x9F]', '', cleaned_chunk_content)

        # 2. Replace common problematic non-breaking spaces or zero-width spaces
        cleaned_chunk_content = cleaned_chunk_content.replace('\u00A0', ' ') # Non-breaking space
        cleaned_chunk_content = cleaned_chunk_content.replace('\u200B', '') # Zero-width space

        # 3. Aggressively remove any other non-printable ASCII control characters (except common ones)
        # This regex keeps letters, numbers, punctuation, and common whitespace (space, tab, newline)
        # This was the aggressive cleaning, now mostly handled before embedding.
        # cleaned_chunk_content = re.sub(r'[^\x20-\x7E\t\n\r]+', '', cleaned_chunk_content)

        # 4. Replace multiple newlines/spaces with single spaces (your original cleaner)
        cleaned_chunk_content = re.sub(r'\s+', ' ', cleaned_chunk_content).strip() # Consolidate whitespace

        print(f"Cleaned Chunk Content: \n{cleaned_chunk_content}")
        # Add the cleaned content to the list for the LLM
        final_retrieved_chunks_for_llm.append(cleaned_chunk_content)

    return final_retrieved_chunks_for_llm # Return the cleaned list for the LLM


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
    

def evaluate_system(qa_pairs, db_collection, embedding_model, llm_function, top_k=3):
    """
    Evaluates the RAG system against a list of question-answer pairs.

    Args:
        qa_pairs (list): A list of dictionaries, each with 'question' and 'answer' (ground truth).
        db_collection: The ChromaDB collection.
        embedding_model: The SentenceTransformer embedding model.
        llm_function: The function that calls your LLM (generate_answer_with_llm).
        top_k (int): Number of chunks to retrieve for the LLM.
    """
   
    print("\n" + "="*80)
    print("                 STARTING SYSTEM EVALUATION                 ")
    print("="*80 + "\n")

    rouge = load("rouge")
    results_log = []

    for i, qa_pair in enumerate(qa_pairs):
        question = qa_pair['question']
        ground_truth_answer = qa_pair['answer']

        print(f"\n--- Evaluation Case {i+1}/{len(qa_pairs)} ---")
        print(f"Question: {question}")
        print(f"Ground Truth: {ground_truth_answer}")

        # 1. Retrieval
        retrieved_context_chunks = retrieve_context(question, db_collection, embedding_model, top_k=top_k)

        # 2. Generation
        system_answer = llm_function(question, retrieved_context_chunks)
        print(f"\nSystem Answer: {system_answer}")

        # 3. Evaluation (ROUGE)
        # ROUGE expects lists of strings for predictions and references
        rouge_results = rouge.compute(
            predictions=[system_answer],
            references=[ground_truth_answer]
        )
        print(f"ROUGE Scores: {rouge_results}")

        results_log.append({
            "question": question,
            "ground_truth": ground_truth_answer,
            "retrieved_chunks": retrieved_context_chunks, # Store retrieved content for debugging
            "system_answer": system_answer,
            "rouge_scores": rouge_results
        })
        print("-" * 50)

    print("\n" + "="*80)
    print("                 EVALUATION COMPLETE                  ")
    print("="*80 + "\n")

    # You can save the results log to a JSON file for later analysis
    with open("evaluation_results.json", "w", encoding="utf-8") as f:
        json.dump(results_log, f, indent=4, ensure_ascii=False)
    print("Evaluation results saved to evaluation_results.json")

    return results_log

# --- Sample QA Dataset for the provided document ---
# Manually created from your '11316-Article Text-14844-1-2-20201228.pdf'
sample_qa_dataset = [
    {
        "question": "What is the primary contribution of the proposed unified model?",
        "answer": "The unified model proposes a human-like reading strategy for DBQA task which is similar to the logic of students when they do the test of reading comprehension, and it combines general understanding of both document and question."
    },
    {
        "question": "What are the three major encoding layers of the unified model?",
        "answer": "The unified model contains three major encoding layers: the basic encoder, combined encoder and hierarchical encoder."
    },
    {
        "question": "On which two datasets were extensive experiments conducted?",
        "answer": "Extensive experiments were conducted on both the English WikiQA dataset and the Chinese dataset (NLPCC2016)."
    },
    {
        "question": "What does DBQA stand for?",
        "answer": "DBQA stands for Document-based Question Answering."
    },
    {
        "question": "What are the three steps of the human-like reading strategy mentioned in the paper?",
        "answer": "The detailed reading strategy is as follows: 1. Go over the document quickly to get a general understanding. 2. Read the question carefully equipped with the general understanding. 3. Go back to the document with the prior knowledge of question and get the right answer."
    },
    {
        "question": "What are the two ways mentioned to automatically summarize a document?",
        "answer": "There are two ways to automatically summarize a document: extractive summarization and abstractive summarization."
    },
    {
        "question": "What are the evaluation metrics used in the experiments?",
        "answer": "The evaluation metrics used are Mean Average Precision (MAP) and Mean Reciprocal Rank (MRR)."
    },
    {
        "question": "Which dataset does not have a natural title for each document?",
        "answer": "The Chinese DBQA dataset (NLPCC2016) does not have a natural title for each document."
    },
    {
        "question": "Who is the corresponding author of this paper?",
        "answer": "The corresponding author of this paper is Yunfang Wu."
    },
    {
        "question": "What is the name of the conference where this paper was presented?",
        "answer": "This paper was presented at The Thirty-Second AAAI Conference on Artificial Intelligence (AAAI-18)."
    },
    # Add a negative example (question not answerable from the document)
    {
        "question": "What is the capital of France?",
        "answer": "I cannot answer this question as the information is not present in the document." # Expected LLM response
    }
]
    

if __name__ == "__main__":
    print("Loading SentenceTransformer model...")
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2') # Small, fast model for quick testing

    pdf_file_path = "11316-Article Text-14844-1-2-20201228.pdf"

if not os.path.exists(pdf_file_path):
    print(f"Error: '{pdf_file_path}' not found. Please place a text-searchable PDF in the script directory.")
    exit()

print(f"--- Processing Document: {pdf_file_path} ---")



# # 1. Document Ingestion & Preprocessing
# full_markdown_text = extract_markdown_from_pdf(pdf_file_path)
# if full_markdown_text is None:
#     print("Failed to extract markdown. Exiting.")
#     exit()


# print("--- Cleaning raw markdown text ---")
# cleaned_full_markdown_text = clean_markdown_text(full_markdown_text)
# print("--- Cleaned Markdown Text (sample written to document_cleaned.txt) ---")
# with open("document_cleaned.txt", "w") as file:
#     file.write(cleaned_full_markdown_text)
# print("--- End Cleaned Markdown Text ---\n")

# Now using markdown_aware_chunk_text
processed_chunks_for_db = markdown_aware_chunk_text(
    #cleaned_full_markdown_text, # Use the cleaned text
    extract_text_from_pdf(pdf_file_path), # Use the extracted text
    max_chunk_tokens=500, # Increased chunk size as Markdown chunks can be longer
    chunk_overlap_tokens=100 # Increased overlap for better context
)

print(f"Document processed into {len(processed_chunks_for_db)} markdown-aware chunks.")



db_collection = create_vector_db(processed_chunks_for_db, embedding_model)
if db_collection is None:
    print("Failed to create vector database. Exiting.")
    exit()

evaluation_results = evaluate_system(
    qa_pairs=sample_qa_dataset,
    db_collection=db_collection,
    embedding_model=embedding_model,
    llm_function=generate_answer_with_llm, # Pass your LLM function
    top_k=3 # You can experiment with this
)