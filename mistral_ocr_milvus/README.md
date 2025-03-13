# Mistral OCR + Milvus Integration

This project demonstrates how to combine Mistral AI's OCR capabilities with Milvus vector database for document understanding and semantic search.

## Features

- **Document OCR Processing**: Extract text from PDFs and images using Mistral's OCR capabilities
- **Vector Storage**: Store document content as vector embeddings in Milvus
- **Semantic Search**: Search for similar documents based on semantic meaning
- **Tool-based Approach**: Uses function calling for document processing and search

## Requirements

- Python 3.11+
- Mistral API key
- Milvus database (local or cloud)

## Installation

1. Install the required packages:

```bash
pip install mistralai pymilvus python-dotenv
```

2. Set up a Milvus instance:
   - Local: Follow the [Milvus installation guide](https://milvus.io/docs/install_standalone-docker.md)
   - Cloud: Use [Zilliz Cloud](https://zilliz.com)

3. Set up your environment:
   - Create a `.env` file with your Mistral API key: `MISTRAL_API_KEY=your_api_key_here`
   - The Milvus URI is configured to `http://localhost:19530` by default

## Usage

Run the script:

```bash
python document_understanding_milvus.py
```

The script provides an interactive chat interface with tool-based OCR capabilities.

### Example Commands

- Process a document: `Could you summarize what this research paper talks about? https://arxiv.org/pdf/2410.07073`
- Process an image: `What is written here: https://jeroen.github.io/images/testocr.png`
- Search for similar documents: `What is stored in the collection "document_ocr"`
- Get statistics: `How many documents have been processed so far?`

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

## Available Tools

The system provides three main tools:
- **open_urls**: Process PDFs and images via OCR
- **search_documents**: Search through previously processed documents
- **get_stats**: Get statistics about stored documents

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