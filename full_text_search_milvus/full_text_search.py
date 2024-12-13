from pymilvus import DataType, Function, FunctionType, MilvusClient

client = MilvusClient(uri="http://localhost:19530")

schema = client.create_schema()

schema.add_field(
    field_name="id", datatype=DataType.INT64, is_primary=True, auto_id=True
)
schema.add_field(
    field_name="text", datatype=DataType.VARCHAR, max_length=1000, enable_analyzer=True
)
schema.add_field(field_name="sparse", datatype=DataType.SPARSE_FLOAT_VECTOR)


bm25_function = Function(
    name="text_bm25_emb",  # Function name
    input_field_names=["text"],  # Name of the VARCHAR field containing raw text data
    output_field_names=[
        "sparse"
    ],  # Name of the SPARSE_FLOAT_VECTOR field reserved to store generated embeddings
    function_type=FunctionType.BM25,
)

schema.add_function(bm25_function)


index_params = client.prepare_index_params()

index_params.add_index(field_name="sparse", index_type="AUTOINDEX", metric_type="BM25")

client.create_collection(
    collection_name="demo", schema=schema, index_params=index_params
)

client.insert(
    "demo",
    [
        {"text": "information retrieval is a field of study."},
        {
            "text": "information retrieval focuses on finding relevant information in large datasets."
        },
        {"text": "data mining and information retrieval overlap in research."},
    ],
)

search_params = {
    "params": {"drop_ratio_search": 0.2},
}

if __name__ == "__main__":
    search_result = client.search(
        collection_name="demo",
        data=["whats the focus of information retrieval?"],
        anns_field="sparse",
        limit=3,
        search_params=search_params,
    )
    print(search_result)
