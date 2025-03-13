import json
import os
import re

from dotenv import load_dotenv
from mistralai import Mistral
from pymilvus import CollectionSchema, DataType, FieldSchema, MilvusClient
from pymilvus.client.types import LoadState

load_dotenv()

api_key = os.getenv("MISTRAL_API_KEY")
client = Mistral(api_key=api_key)
text_model = "mistral-small-latest"
ocr_model = "mistral-ocr-latest"
embedding_model = "mistral-embed"
milvus_client = MilvusClient(uri="http://localhost:19530")

# Milvus collection name
COLLECTION_NAME = "document_ocr"

# Define system prompts
SYSTEM_PROMPT_TOOL = """You are an AI Assistant with document understanding via URLs. You will be provided with URLs, and you must answer any questions related to those documents.

# OPEN URLS INSTRUCTIONS
You can open URLs by using the `open_urls` tool. It will open webpages and apply OCR to them, retrieving the contents. Use those contents to answer the user.
Only URLs pointing to PDFs and images are supported; you may encounter an error if they are not; provide that information to the user if required.

# SEARCH DOCUMENTS INSTRUCTIONS
You can search through previously processed documents using the `search_documents` tool. This will perform a semantic search through all stored document content and return the most relevant results.
When presenting search results to the user, always include the source URL and page number for each result.

# DOCUMENT STATISTICS
You can get statistics about stored documents using the `get_stats` tool. This will show how many documents have been processed and stored in the system."""

SYSTEM_PROMPT_BUILTIN = "You are an AI Assistant with document understanding via URLs. You may be provided with URLs, followed by their corresponding OCR."


# Setup Milvus collection
def setup_milvus_collection():
    """Create Milvus collection if it doesn't exist."""

    # Define collection schema
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="url", dtype=DataType.VARCHAR, max_length=500),
        FieldSchema(name="page_num", dtype=DataType.INT64),
        FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=65535),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=1024),
    ]

    schema = CollectionSchema(fields=fields)

    # Create collection
    milvus_client.create_collection(
        collection_name=COLLECTION_NAME,
        schema=schema,
    )
    
    index_params = milvus_client.prepare_index_params()

    index_params.add_index(
        field_name="embedding",
        index_type="IVF_FLAT",
        metric_type="COSINE",
        params={"nlist": 128}
    )
    
    milvus_client.create_index(
        collection_name=COLLECTION_NAME,
        index_params=index_params
    )

    print(f"Collection '{COLLECTION_NAME}' created successfully with index.")


# Generate embeddings using Mistral
def generate_embedding(text):
    """Generate embedding for text using Mistral embedding model."""
    response = client.embeddings.create(
        model=embedding_model,
        inputs=[text]
    )
    return response.data[0].embedding


# Store OCR results in Milvus
def store_ocr_in_milvus(url, ocr_result):
    """Process OCR results and store in Milvus."""
    # Extract pages from OCR result
    pages = []
    current_page = ""
    page_num = 0

    for line in ocr_result.split("\n"):
        if line.startswith("### Page "):
            if current_page:
                pages.append((page_num, current_page.strip()))
            page_num = int(line.replace("### Page ", ""))
            current_page = ""
        else:
            current_page += line + "\n"

    # Add the last page
    if current_page:
        pages.append((page_num, current_page.strip()))

    # Prepare data for Milvus
    entities = []
    for page_num, content in pages:
        # Generate embedding for the page content
        embedding = generate_embedding(content)

        # Create entity
        entity = {
            "url": url,
            "page_num": page_num,
            "content": content,
            "embedding": embedding,
        }
        entities.append(entity)

    # Insert into Milvus
    if entities:
        milvus_client.insert(collection_name=COLLECTION_NAME, data=entities)
        print(f"Stored {len(entities)} pages from {url} in Milvus.")

    return len(entities)


# Define OCR function
def perform_ocr(url):
    """Apply OCR to a URL (PDF or image)."""
    try:
        # Try PDF OCR first
        response = client.ocr.process(
            model=ocr_model, document={"type": "document_url", "document_url": url}
        )
    except Exception:
        try:
            # If PDF OCR fails, try Image OCR
            response = client.ocr.process(
                model=ocr_model, document={"type": "image_url", "image_url": url}
            )
        except Exception as e:
            return str(e)  # Return error message

    # Format the OCR results
    ocr_result = "\n\n".join(
        [
            f"### Page {i + 1}\n{response.pages[i].markdown}"
            for i in range(len(response.pages))
        ]
    )

    # Store in Milvus
    store_ocr_in_milvus(url, ocr_result)

    return ocr_result


# Define tool function
def open_urls(urls):
    """Process a list of URLs and return their contents."""
    contents = "# Documents"
    for url in urls:
        contents += f"\n\n## URL: {url}\n{perform_ocr(url)}"
    return contents


# Get statistics about stored documents
def get_document_stats():
    """Get statistics about documents stored in Milvus."""
    if not milvus_client.has_collection(COLLECTION_NAME):
        return "No documents have been processed yet."

    # Get collection stats
    stats = milvus_client.get_collection_stats(COLLECTION_NAME)
    row_count = stats["row_count"]

    # Get unique URLs
    results = milvus_client.query(
        collection_name=COLLECTION_NAME, filter="", output_fields=["url"], limit=10000
    )

    unique_urls = set()
    for result in results:
        unique_urls.add(result["url"])

    # Format stats
    stats_text = "# Document Statistics\n\n"
    stats_text += f"Total pages stored: {row_count}\n"
    stats_text += f"Unique documents: {len(unique_urls)}\n\n"

    if unique_urls:
        stats_text += "## Processed Documents:\n"
        for i, url in enumerate(unique_urls):
            stats_text += f"{i + 1}. {url}\n"

    return stats_text


# URL extraction function
def extract_urls(text):
    """Extract URLs from text using regex."""
    url_pattern = r"\b((?:https?|ftp)://(?:www\.)?[^\s/$.?#].[^\s]*)\b"
    return re.findall(url_pattern, text)


# Search Milvus for similar content
def search_milvus(query, limit=5):
    """Search Milvus for similar content to the query."""
    # Check if collection exists
    if not milvus_client.has_collection(COLLECTION_NAME):
        return "No documents have been processed yet."
    
    # Load collection if not already loaded
    if milvus_client.get_load_state(COLLECTION_NAME) != LoadState.Loaded:
        milvus_client.load_collection(COLLECTION_NAME)
    
    print(f"Searching Milvus for query: {query}")
    query_embedding = generate_embedding(query)

    search_params = {
        "metric_type": "COSINE",
        "params": {"nprobe": 10}
    }

    search_results = milvus_client.search(
        collection_name=COLLECTION_NAME,
        data=[query_embedding],
        anns_field="embedding",
        search_params=search_params,
        limit=limit,
        output_fields=["url", "page_num", "content"],
    )

    formatted_results = "# Search Results\n\n"

    if not search_results:
        return "No matching documents found."

    for i, hit in enumerate(search_results[0]):
        print(f"Hit: {hit}, type: {type(hit)}")
        print(f"Hit keys: {hit.keys()}, type: {type(hit.keys())}")
        print(f"Hit entity: {hit['entity']}, type: {type(hit['entity'])}")

        url = hit['entity']['url']
        page_num = hit['entity']['page_num']
        content = hit['entity']['content']
        score = hit['distance']

        formatted_results += f"## Result {i + 1} (Score: {score:.2f})\n"
        formatted_results += f"**Source:** {url} (Page {page_num})\n\n"

        # Truncate content if too long
        if len(content) > 500:
            content = content[:500] + "..."

        formatted_results += f"{content}\n\n"

    return formatted_results


# Define tool schema
tools = [
    {
        "type": "function",
        "function": {
            "name": "open_urls",
            "description": "Open URLs websites (PDFs and Images) and perform OCR on them.",
            "parameters": {
                "type": "object",
                "properties": {
                    "urls": {
                        "type": "array",
                        "description": "The URLs list.",
                    }
                },
                "required": ["urls"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_documents",
            "description": "Search through previously processed documents using semantic search with Milvus.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query to find relevant document content.",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of results to return (default: 5).",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_stats",
            "description": "Get statistics about documents stored in the system.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
]

# Map function names to actual functions
tool_functions = {
    "open_urls": open_urls,
    "search_documents": search_milvus,
    "get_stats": get_document_stats,
}


# Chat loop using tool-based approach
def chat_with_tools():
    """Interactive chat loop using the tool-based approach."""
    if not milvus_client.has_collection(COLLECTION_NAME):
        setup_milvus_collection()

    messages = [{"role": "system", "content": SYSTEM_PROMPT_TOOL}]

    print("Chat with Tool-based OCR (type 'quit' to exit)")
    while True:
        user_input = input("User > ")
        if user_input.lower() == "quit":
            break

        messages.append({"role": "user", "content": user_input})

        # Loop until no more tool calls
        while True:
            response = client.chat.complete(
                model=text_model, messages=messages, temperature=0, tools=tools
            )
            assistant_message = response.choices[0].message
            messages.append(
                {
                    "role": "assistant",
                    "content": assistant_message.content,
                    "tool_calls": assistant_message.tool_calls,
                }
            )

            # If no tool calls, break the loop
            if not assistant_message.tool_calls:
                break

            # Process tool calls
            tool_call = assistant_message.tool_calls[0]
            function_name = tool_call.function.name
            function_params = json.loads(tool_call.function.arguments)
            function_result = tool_functions[function_name](**function_params)

            messages.append(
                {
                    "role": "tool",
                    "name": function_name,
                    "content": function_result,
                    "tool_call_id": tool_call.id,
                }
            )

        print("Assistant >", assistant_message.content)


if __name__ == "__main__":
    chat_with_tools()