# ColPali + Milvus: Revolutionizing Document Retrieval with Vision-Language Models

In the world of document retrieval and search, we've traditionally relied on complex pipelines that extract text, analyze layouts, and generate embeddings. But what if we could skip all that and simply "see" documents the way humans do? Enter **ColPali** - a groundbreaking approach that combines the power of vision-language models with efficient retrieval techniques to transform how we search through documents.

## The Problem with Traditional Document Retrieval

Traditional document retrieval systems, especially for PDFs and other rich documents, require a cumbersome pipeline:

1. Run OCR to extract text from scanned documents
2. Perform layout detection to identify paragraphs, figures, and tables
3. Reconstruct the document structure and reading order
4. Caption figures and tables using specialized models
5. Chunk text into manageable pieces
6. Generate embeddings for each chunk
7. Store these embeddings in a vector database

This process is slow, error-prone, and often fails to capture the rich visual information in documents. Tables, figures, and layout information get lost or distorted, leading to suboptimal search results.

## Enter ColPali: What You See Is What You Search

ColPali (Contextualized Late Interaction over PaliGemma) takes a radically different approach. Instead of extracting text, it treats document pages as images and leverages vision-language models to understand both textual and visual content simultaneously.

The name "ColPali" comes from two key components:
- **Col**: Refers to ColBERT's multi-vector representation and late interaction strategy
- **Pali**: Refers to PaliGemma, Google's vision-language model that combines SigLIP-So400m (image encoder) and Gemma-2B (text decoder)

### How ColPali Works

ColPali's architecture is elegant in its simplicity:

1. **During indexing**: Document pages are converted to images, and a vision-language model (PaliGemma) processes these images to generate a grid of contextualized embeddings (32×32 patches, each represented as a 128-dimensional vector).

2. **During querying**: The user's query is tokenized and embedded. A "late interaction" mechanism compares each query token with all document patches to find the most relevant matches.

The magic happens in this late interaction - for each term in the query, ColPali finds the document patch with the most similar representation, then sums these similarity scores to produce a final relevance score. This allows for rich interaction between query terms and document elements while maintaining efficient retrieval.

## Implementing ColPali with Milvus

[Milvus](https://milvus.io/), a powerful vector database, is the perfect companion for ColPali's multi-vector approach. Let's see how we can implement this powerful combination.

### Setting Up the Environment

First, we need to install the necessary packages:

```python
pip install pdf2image pymilvus colpali_engine tqdm pillow
```

### Converting PDFs to Images

Since ColPali works with images, we need to convert our PDF documents to image format:

```python
from pdf2image import convert_from_path

pdf_path = "pdfs/your_document.pdf"
images = convert_from_path(pdf_path)

for i, image in enumerate(images):
    image.save(f"pages/page_{i + 1}.png", "PNG")
```

### Creating a Milvus Collection for ColPali Embeddings

Now, let's set up a Milvus collection to store our ColPali embeddings:

```python
from pymilvus import MilvusClient, DataType
import numpy as np
import concurrent.futures

# Initialize Milvus client
client = MilvusClient(uri="milvus.db")  # For local testing with Milvus Lite
# For production: client = MilvusClient(uri="http://your-milvus-server:19530")
```

We'll create a custom retriever class to handle the multi-vector nature of ColPali embeddings:

```python
class MilvusColbertRetriever:
    def __init__(self, milvus_client, collection_name, dim=128):
        self.collection_name = collection_name
        self.client = milvus_client
        if self.client.has_collection(collection_name=self.collection_name):
            self.client.load_collection(collection_name)
        self.dim = dim

    def create_collection(self):
        if self.client.has_collection(collection_name=self.collection_name):
            self.client.drop_collection(collection_name=self.collection_name)
        
        schema = self.client.create_schema(
            auto_id=True,
            enable_dynamic_fields=True,
        )
        schema.add_field(field_name="pk", datatype=DataType.INT64, is_primary=True)
        schema.add_field(
            field_name="vector", datatype=DataType.FLOAT_VECTOR, dim=self.dim
        )
        schema.add_field(field_name="seq_id", datatype=DataType.INT16)
        schema.add_field(field_name="doc_id", datatype=DataType.INT64)
        schema.add_field(field_name="doc", datatype=DataType.VARCHAR, max_length=65535)

        self.client.create_collection(
            collection_name=self.collection_name, schema=schema
        )

    def create_index(self):
        self.client.release_collection(collection_name=self.collection_name)
        self.client.drop_index(
            collection_name=self.collection_name, index_name="vector"
        )
        index_params = self.client.prepare_index_params()
        index_params.add_index(
            field_name="vector",
            index_name="vector_index",
            index_type="HNSW",
            metric_type="IP",
            params={
                "M": 16,
                "efConstruction": 500,
            },
        )

        self.client.create_index(
            collection_name=self.collection_name, index_params=index_params, sync=True
        )
```

### The Search Magic: Implementing Late Interaction

The heart of ColPali is its search mechanism, which implements the late interaction approach:

```python
def search(self, data, topk):
    # First, perform a vector search to find candidate documents
    search_params = {"metric_type": "IP", "params": {}}
    results = self.client.search(
        self.collection_name,
        data,
        limit=int(50),
        output_fields=["vector", "seq_id", "doc_id"],
        search_params=search_params,
    )
    
    # Collect unique document IDs from the results
    doc_ids = set()
    for r_id in range(len(results)):
        for r in range(len(results[r_id])):
            doc_ids.add(results[r_id][r]["entity"]["doc_id"])

    scores = []

    # Rerank function to calculate MaxSim score for each document
    def rerank_single_doc(doc_id, data, client, collection_name):
        # Retrieve all embeddings for this document
        doc_colbert_vecs = client.query(
            collection_name=collection_name,
            filter=f"doc_id in [{doc_id}]",
            output_fields=["seq_id", "vector", "doc"],
            limit=1000,
        )
        
        # Stack all vectors for this document
        doc_vecs = np.vstack(
            [doc_colbert_vecs[i]["vector"] for i in range(len(doc_colbert_vecs))]
        )
        
        # Calculate MaxSim score: for each query token, find the most similar document token
        # and sum these maximum similarities
        score = np.dot(data, doc_vecs.T).max(1).sum()
        return (score, doc_id)

    # Use parallel processing to rerank documents
    with concurrent.futures.ThreadPoolExecutor(max_workers=300) as executor:
        futures = {
            executor.submit(
                rerank_single_doc, doc_id, data, client, self.collection_name
            ): doc_id
            for doc_id in doc_ids
        }
        for future in concurrent.futures.as_completed(futures):
            score, doc_id = future.result()
            scores.append((score, doc_id))

    # Sort by score and return top-k results
    scores.sort(key=lambda x: x[0], reverse=True)
    if len(scores) >= topk:
        return scores[:topk]
    else:
        return scores
```

This implementation follows the MaxSim operation described in the ColBERT paper: for each query token, find the document token with the highest similarity, then sum these maximum similarities to get the final score.

### Generating and Storing ColPali Embeddings

Now, let's use the ColPali model to generate embeddings for our documents:

```python
from colpali_engine.models import ColPali
from colpali_engine.models.paligemma.colpali.processing_colpali import ColPaliProcessor
from colpali_engine.utils.torch_utils import ListDataset, get_torch_device
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm
from PIL import Image
import os

# Initialize the ColPali model
device = get_torch_device("cpu")  # Use GPU if available
model_name = "vidore/colpali-v1.2"

model = ColPali.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map=device,
).eval()

processor = ColPaliProcessor.from_pretrained(model_name)

# Process document images
images = [Image.open(f"./pages/{name}") for name in os.listdir("./pages")]

dataloader = DataLoader(
    dataset=ListDataset[str](images),
    batch_size=1,
    shuffle=False,
    collate_fn=lambda x: processor.process_images(x),
)

document_embeddings = []
for batch_doc in tqdm(dataloader):
    with torch.no_grad():
        batch_doc = {k: v.to(model.device) for k, v in batch_doc.items()}
        embeddings_doc = model(**batch_doc)
    document_embeddings.extend(list(torch.unbind(embeddings_doc.to("cpu"))))

# Create and set up the Milvus collection
retriever = MilvusColbertRetriever(collection_name="colpali", milvus_client=client)
retriever.create_collection()
retriever.create_index()

# Insert embeddings into Milvus
filepaths = [f"./pages/{name}" for name in os.listdir("./pages")]
for i in range(len(filepaths)):
    data = {
        "colbert_vecs": document_embeddings[i].float().numpy(),
        "doc_id": i,
        "filepath": filepaths[i],
    }
    retriever.insert(data)
```

### Searching with ColPali

Finally, let's see how we can search for relevant documents using ColPali:

```python
# Process queries
queries = [
    "How does ColBERT perform end-to-end retrieval?",
    "Show me the performance comparison table for ColBERT",
]

dataloader = DataLoader(
    dataset=ListDataset[str](queries),
    batch_size=1,
    shuffle=False,
    collate_fn=lambda x: processor.process_queries(x),
)

query_embeddings = []
for batch_query in dataloader:
    with torch.no_grad():
        batch_query = {k: v.to(model.device) for k, v in batch_query.items()}
        embeddings_query = model(**batch_query)
    query_embeddings.extend(list(torch.unbind(embeddings_query.to("cpu"))))

# Search for each query
for i, query in enumerate(queries):
    query_embedding = query_embeddings[i].float().numpy()
    results = retriever.search(query_embedding, topk=3)
    
    print(f"Query: {query}")
    for score, doc_id in results:
        print(f"  Score: {score:.4f}, Document: {filepaths[doc_id]}")
    print()
```

## Why ColPali + Milvus is a Game-Changer

The combination of ColPali and Milvus offers several compelling advantages:

1. **Simplified Pipeline**: No more complex text extraction, OCR, layout analysis, or chunking. Just convert pages to images and process them directly.

2. **Rich Visual Understanding**: ColPali captures both textual and visual information, including tables, figures, and layout, leading to more comprehensive document understanding.

3. **Superior Performance**: ColPali outperforms traditional text-based retrieval methods, especially for visually complex documents containing infographics, figures, and tables.

4. **Efficient Storage and Retrieval**: Milvus provides fast and scalable vector search capabilities, making it ideal for storing and retrieving ColPali's multi-vector representations.

5. **Interpretable Results**: ColPali can visualize which parts of a document match specific query terms, providing insights into why a document was retrieved.

## Real-World Applications

The ColPali + Milvus combination is particularly valuable for:

- **Legal Document Search**: Find relevant cases and precedents in legal documents with complex layouts and tables.
- **Scientific Literature Review**: Search through research papers with figures, equations, and tables.
- **Technical Documentation**: Navigate complex technical manuals with diagrams and schematics.
- **Financial Analysis**: Search through reports with charts, graphs, and financial tables.

## Conclusion

ColPali represents a paradigm shift in document retrieval - moving from "what you extract is what you search" to "what you see is what you search." By leveraging vision-language models and multi-vector representations, it offers a more intuitive and effective way to search through documents.

When combined with Milvus's powerful vector search capabilities, ColPali becomes a practical solution for real-world document retrieval challenges. The simplified pipeline, improved accuracy, and ability to understand visual elements make this combination a compelling choice for modern retrieval-augmented generation systems.

As vision-language models continue to evolve, we can expect even more powerful document understanding capabilities in the future. But for now, ColPali + Milvus represents the state-of-the-art in document retrieval - a solution that truly sees documents the way humans do.

## Getting Started

Ready to try ColPali with Milvus? Check out the [complete code example](https://github.com/milvus-io/bootcamp/blob/master/bootcamp/tutorials/quickstart/use_ColPali_with_milvus.ipynb) on GitHub, or visit the [ColPali repository](https://github.com/illuin-tech/colpali) to learn more about the model.

For more information about Milvus, visit the [official documentation](https://milvus.io/docs). 