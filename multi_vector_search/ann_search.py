from pymilvus import AnnSearchRequest, RRFRanker, Collection, connections

query_filmVector = [
    [
        0.8896863042430693,
        0.370613100114602,
        0.23779315077113428,
        0.38227915951132996,
        0.5997064603128835,
    ]
]

search_param_1 = {
    "data": query_filmVector,  # Query vector
    "anns_field": "filmVector",  # Vector field name
    "param": {
        "metric_type": "L2",  # This parameter value must be identical to the one used in the collection schema
        "params": {"nprobe": 10},
    },
    "limit": 2,  # Number of search results to return in this AnnSearchRequest
}
request_1 = AnnSearchRequest(**search_param_1)

query_posterVector = [
    [
        0.02550758562349764,
        0.006085637357292062,
        0.5325251250159071,
        0.7676432650114147,
        0.5521074424751443,
    ]
]
search_param_2 = {
    "data": query_posterVector,  # Query vector
    "anns_field": "posterVector",  # Vector field name
    "param": {
        "metric_type": "L2",  # This parameter value must be identical to the one used in the collection schema
        "params": {"nprobe": 10},
    },
    "limit": 2,  # Number of search results to return in this AnnSearchRequest
}
request_2 = AnnSearchRequest(**search_param_2)

reqs = [request_1, request_2]

rerank = RRFRanker()


connections.connect(
    host="0.0.0.0",
    port="19530",
)

collection = Collection(name="test_collection")
collection.load()

from pymilvus import MilvusClient

client = MilvusClient()

res = collection.hybrid_search(
    reqs,  # List of AnnSearchRequests created in step 1
    rerank,  # Reranking strategy specified in step 2
    limit=2,  # Number of final search results to return
)

print(res)
