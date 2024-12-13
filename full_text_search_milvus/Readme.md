# Milvus Full-Text Search Demo

Simple demo showing how to use Milvus 2.5's built-in full-text search capabilities without needing a separate search engine like Elasticsearch.

## What it Does
- Sets up a Milvus collection that handles both text and sparse vectors
- Automatically converts text to BM25 sparse vectors under the hood
- Demonstrates simple keyword search without manual vector generation

## Key Features
- Single system for both keyword and vector search
- No embedding pipeline needed
- Automatic BM25 conversion
- Real-time search statistics updates


## Prerequisites
- Milvus 2.5 or higher

## Benefits
- Eliminates need for separate Elasticsearch deployment
- Simplified architecture and maintenance
- Native handling of technical terms
- Combined exact and semantic matching capabilities