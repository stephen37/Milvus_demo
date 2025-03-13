# Mistral OCR + Milvus Integration

This project demonstrates how to combine Mistral AI's OCR capabilities with Milvus vector database for document understanding and semantic search.

## Features

- **Document OCR Processing**: Extract text from PDFs and images using Mistral's OCR capabilities
- **Vector Storage**: Store document content as vector embeddings in Milvus
- **Semantic Search**: Search for similar documents based on semantic meaning
- **Two Operation Modes**:
  - Tool-based approach with function calling
  - Built-in OCR approach with direct document processing

## Requirements

- Python 3.8+
- Mistral API key
- Milvus database (local or cloud)

## Installation

1. Install the required packages:

```bash
pip install mistralai pymilvus numpy
```

2. Set up a Milvus instance:
   - Local: Follow the [Milvus installation guide](https://milvus.io/docs/install_standalone-docker.md)
   - Cloud: Use Zilliz Cloud or other Milvus cloud providers

3. Update the configuration in `document_understanding.py`:
   - Set your Mistral API key
   - Configure the Milvus URI (default: `http://localhost:19530`)

## Usage

Run the script:

```bash
python document_understanding.py
```

Choose between two demo modes:
1. **Tool Usage Demo**: Uses function calling to process documents and search
2. **Built-in OCR Demo**: Uses Mistral's built-in OCR capabilities

### Example Commands

- Process a document: `Could you summarize what this research paper talks about? https://arxiv.org/pdf/2410.07073`
- Process an image: `What is written here: https://jeroen.github.io/images/testocr.png`
- Search for similar documents: `Find documents similar to: vector databases for document retrieval`

## How It Works

1. **Document Processing**:
   - Documents are processed using Mistral's OCR capabilities
   - Text is extracted and formatted as markdown

2. **Vector Storage**:
   - Document content is converted to embeddings using Mistral's embedding model
   - Embeddings are stored in Milvus along with metadata (URL, page number, content)

3. **Semantic Search**:
   - User queries are converted to embeddings
   - Milvus performs similarity search to find relevant documents
   - Results are returned with relevance scores

## Architecture

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│  Document   │    │   Mistral   │    │   Milvus    │
│    URL      │───>│     OCR     │───>│  Database   │
└─────────────┘    └─────────────┘    └─────────────┘
                          │                  │
                          ▼                  ▼
                   ┌─────────────┐    ┌─────────────┐
                   │   Mistral   │    │  Semantic   │
                   │     LLM     │<───│   Search    │
                   └─────────────┘    └─────────────┘
                          │
                          ▼
                   ┌─────────────┐
                   │    User     │
                   │  Response   │
                   └─────────────┘
```
