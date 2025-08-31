# **RAG-Based Document Question Answering System**

## **Overview**

This repository contains a Python-based Retrieval Augmented Generation (RAG) system for document-based question answering. It was developed as a one-week, self-directed research project for a university AI lab application.

The project demonstrates a practical implementation of a modern RAG pipeline designed to answer natural language questions based on the content of a collection of private PDF documents.

## **Key Features**

* **Robust Document Parsing:** A multi-stage approach that first attempts to extract text from PDFs and then falls back to OCR if a document is image-based. This system is designed to handle both text-based and scanned PDFs.  
* **Intelligent Text Cleaning:** A custom function to clean up common artifacts from OCR and PDF text extraction, such as hyphenation and non-standard characters, ensuring the text is ready for processing.  
* **Semantic Chunking:** Employs a strategy that respects sentence boundaries and adds overlap for improved contextual awareness, which is vital for effective retrieval without exceeding the LLM's context window.  
* **Multi-Document Processing:** The system is built to ingest and process a collection of multiple PDF documents, creating a unified knowledge base. It uses concurrent processing for faster ingestion.  
* **Vector-Based Retrieval:** Uses a Sentence-Transformer model (all-MiniLM-L6-v2) to create semantic embeddings and ChromaDB for efficient vector similarity search, which is configured to use cosine similarity.  
* **LLM Integration:** Augments a Google Gemini API call with retrieved document context to generate accurate and grounded answers. The prompt is engineered to instruct the model to use the retrieved context exclusively.  
* **Systematic Evaluation:** Includes a methodology for quantitatively and qualitatively evaluating performance on a custom test set, using metrics like ROUGE scores to assess the quality of the generated answers.

## **The Problem & The Solution**

The goal of this project was to build a system that could "read" a given document and answer questions about its content.

Traditional Large Language Models (LLMs), without external context, are susceptible to **hallucination** (making up facts) and are limited by their knowledge cut-off date. Our solution uses a RAG pipeline to overcome these limitations by:

1. **Retrieval:** Finding the most relevant information from a document or documents.  
2. **Augmentation:** Feeding that information into the LLM's prompt.  
3. **Generation:** Producing an answer grounded in the provided facts.

## **Project Architecture & Components**

The system is a modular RAG pipeline built with standard, open-source libraries.

* **Document Loader:** The multi-doc.py script uses a robust approach that first attempts direct text extraction from PDFs using PyMuPDF. If this fails (e.g., for scanned documents), it falls back to a pipeline using pdf2image and pytesseract for Optical Character Recognition (OCR).  
* **Chunking Strategy:** The custom chunk\_plain\_text function respects semantic boundaries by using NLTK to tokenize text into sentences before grouping them into chunks with a specified overlap.  
* **Embedding Model:** sentence-transformers/all-MiniLM-L6-v2 is used for its high performance and small size, which is ideal for a local development environment.  
* **Vector Store:** ChromaDB is used for its ease of use in a local-first setup and its ability to store and filter metadata alongside embeddings.  
* **Generative Model:** Google Gemini API (gemini-1.5-pro) is used to generate fluent, human-like responses based on the retrieved context.

## **How to Run the Project**

### **Prerequisites**

* Python 3.8+  
* A Google Gemini API key (set as an environment variable GEMINI\_API\_KEY).  
* Tesseract OCR engine installed and configured in your system's PATH.  
* A directory named sample\_data containing one or more PDF documents in the project root.  
* NLTK's punkt tokenizer data.

### **Installation**

1. Clone the repository:
   ```
   git clone \[https://github.com/your-username/your-repo-name.git\](https://github.com/your-username/your-repo-name.git)  
   cd your-repo-name
   ```

2. Install the required dependencies:
   ```
   pip install \-r requirements.txt
   ```
   (Note: You will need to create a requirements.txt file by running pip freeze \> requirements.txt after installing the dependencies.)
   
4. Download NLTK data:
   ``` 
   python \-c "import nltk; nltk.download('punkt')"
   ```

### **Usage**

Run the main script from your terminal:
```
python multi-doc.py
```
The system will process all PDFs in the sample\_data directory, create a vector store, run the evaluation on a pre-defined dataset, and save the results to evaluation\_results.json.

## **Evaluation & Performance**

A key part of this project was designing a method to measure the system's performance objectively. The included evaluation script runs the system on a pre-defined set of questions and compares the generated answers to a set of ground truth answers. The script outputs a JSON file with the results, including ROUGE scores, for each question.

## **Evaluation Data**

The sample\_data directory contains the following academic papers that were used for evaluation:

* **11316-Article Text-14844-1-2-20201228.pdf**: A research paper on Document-based Question Answering (DBQA).  
* **2404.07221v2.pdf**: A paper on optimizing Retrieval for RAG models.  
* **SHTI-316-SHTI240567.pdf**: A study on data extraction from German medical documents.

## **References**

The following papers and resources were used as a basis for this project:

* **Reimers, N., & Gurevych, I. (2019).** *Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks.* A foundational paper on sentence embeddings.  
* **Lewis, P., et al. (2020).** *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks.* The original paper on the RAG paradigm.  
* **sentence-transformers library documentation.**  
* **chromadb library documentation.**  
* **pdf2image and pytesseract documentation.**  
* **google.generativeai library documentation.**

