from typing import Any, Dict

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_core.runnables import RunnableLambda
from langserve import add_routes
from langgraph.graph import StateGraph
from langgraph.graph.graph import END
from typing_extensions import TypedDict

from langchain_community.chat_models import ChatOllama
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_milvus import Milvus
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.embeddings import Embeddings


def load_and_split_documents(urls: list[str]) -> list[Document]:
    docs = [WebBaseLoader(url).load() for url in urls]
    docs_list = [item for sublist in docs for item in sublist]
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=250, chunk_overlap=0
    )
    return text_splitter.split_documents(docs_list)


def add_documents_to_milvus(
    doc_splits: list[Document], embedding_model: Embeddings, connection_args: Any
):
    vectorstore = Milvus.from_documents(
        documents=doc_splits,
        collection_name="rag_milvus",
        embedding=embedding_model,
        connection_args=connection_args,
    )
    return vectorstore.as_retriever()


# Initialize the components
urls = [
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
    "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
    "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
]

doc_splits = load_and_split_documents(urls)
embedding_model = HuggingFaceEmbeddings()
connection_args = {"uri": "./milvus_rag.db"}
retriever = add_documents_to_milvus(doc_splits, embedding_model, connection_args)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant."),
        ("user", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
)

retrieval_grader_prompt = PromptTemplate(
    template="""You are a grader assessing relevance of a retrieved document to a user question. If the document contains keywords related to the user question, grade it as relevant. It does not need to be a stringent test. The goal is to filter out erroneous retrievals. 
    Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question.
    Provide the binary score as a JSON with a single key 'score' and no preamble or explanation.
    Here is the retrieved document: 
    {document}
    Here is the user question: 
    {question}""",
    input_variables=["question", "document"],
)

answer_prompt = PromptTemplate(
    template="""You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. 
    Use three sentences maximum and keep the answer concise:
    Question: {question} 
    Context: {context} 
    Answer:""",
    input_variables=["question", "context"],
)

hallucination_grader_prompt = PromptTemplate(
    template="""You are a grader assessing whether an answer is grounded in / supported by a set of facts. Give a binary score 'yes' or 'no' score to indicate whether the answer is grounded in / supported by a set of facts. Provide the binary score as a JSON with a single key 'score' and no preamble or explanation.
    Here are the facts:
    {documents} 
    Here is the answer: 
    {generation}""",
    input_variables=["generation", "documents"],
)

question_router_prompt = PromptTemplate(
    template="""You are an expert at routing a user question to a vectorstore or web search. Use the vectorstore for questions on LLM agents, prompt engineering, and adversarial attacks. You do not need to be stringent with the keywords in the question related to these topics. Otherwise, use web-search. Give a binary choice 'web_search' or 'vectorstore' based on the question. Return the a JSON with a single key 'datasource' and no preamble or explanation. 
    Question to route: 
    {question}""",
    input_variables=["question"],
)

local_llm = "llama3"
llm_json = ChatOllama(model=local_llm, format="json", temperature=0)
llm_str = ChatOllama(model=local_llm, temperature=0)

retrieval_grader = retrieval_grader_prompt | llm_json | JsonOutputParser()
hallucination_grader = hallucination_grader_prompt | llm_json | JsonOutputParser()
question_router = question_router_prompt | llm_json | JsonOutputParser()


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


rag_chain = answer_prompt | llm_str | StrOutputParser()


def retrieve(state: Dict[str, Any]) -> Dict[str, Any]:
    print("---RETRIEVE---")
    question = state["question"]
    documents = retriever.invoke(question)
    return {"documents": [doc.page_content for doc in documents], "question": question}


def generate(state: Dict[str, Any]) -> Dict[str, Any]:
    print("---GENERATE---")
    question = state["question"]
    documents = state["documents"]
    generation = rag_chain.invoke(
        {"context": "\n\n".join(documents), "question": question}
    )
    return {"documents": documents, "question": question, "generation": generation}


def grade_documents(state: Dict[str, Any]) -> Dict[str, Any]:
    print("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
    question = state["question"]
    documents = state["documents"]
    filtered_docs = []
    web_search = "No"
    for doc in documents:
        score = retrieval_grader.invoke({"question": question, "document": doc})
        if score["score"].lower() == "yes":
            print("---GRADE: DOCUMENT RELEVANT---")
            filtered_docs.append(doc)
        else:
            print("---GRADE: DOCUMENT NOT RELEVANT---")
            web_search = "Yes"
    return {"documents": filtered_docs, "question": question, "web_search": web_search}


def web_search(state: Dict[str, Any]) -> Dict[str, Any]:
    print("---WEB SEARCH---")
    question = state["question"]
    documents = state["documents"]
    docs = TavilySearchResults(k=3).invoke({"query": question})
    web_results = "\n".join([d["content"] for d in docs])
    documents.append(web_results)
    return {"documents": documents, "question": question}


def route_question(state: Dict[str, Any]) -> str:
    print("---ROUTE QUESTION---")
    question = state["question"]
    source = question_router.invoke({"question": question})
    return "websearch" if source["datasource"] == "web_search" else "vectorstore"


def decide_to_generate(state: Dict[str, Any]) -> str:
    print("---ASSESS GRADED DOCUMENTS---")
    return "websearch" if state["web_search"] == "Yes" else "generate"


def grade_generation_v_documents_and_question(state: Dict[str, Any]) -> str:
    print("---CHECK HALLUCINATIONS---")
    score = hallucination_grader.invoke(
        {"documents": state["documents"], "generation": state["generation"]}
    )
    return "useful" if score["score"] == "yes" else "not supported"


# Define Pydantic model for request body
class QuestionRequest(BaseModel):
    question: str


# Initialize FastAPI app
fastapi_app = FastAPI()


# Define LangServe route for text analysis
@fastapi_app.post("/analyze")
async def analyze_text(request: QuestionRequest):
    # Simulate text analysis (replace with your actual LangServe logic)
    entities = ["entity1", "entity2"]
    processed_data = f"Processed entities: {entities}"
    return {"entities": entities, "processed_data": processed_data}


# Add LangServe routes to the app
add_routes(fastapi_app, RunnableLambda(analyze_text))


class GraphState(TypedDict):
    question: str
    generation: str
    web_search: str
    documents: list[str]


# Define the LangGraph workflow
workflow = StateGraph(GraphState)
workflow.add_node("websearch", web_search)
workflow.add_node("retrieve", retrieve)
workflow.add_node("grade_documents", grade_documents)
workflow.add_node("generate", generate)

workflow.set_conditional_entry_point(
    route_question,
    {
        "websearch": "websearch",
        "vectorstore": "retrieve",
    },
)

workflow.add_edge("retrieve", "grade_documents")
workflow.add_conditional_edges(
    "grade_documents",
    decide_to_generate,
    {
        "websearch": "websearch",
        "generate": "generate",
    },
)
workflow.add_edge("websearch", "generate")
workflow.add_conditional_edges(
    "generate",
    grade_generation_v_documents_and_question,
    {
        "not supported": "generate",
        "useful": END,
        "not useful": "websearch",
    },
)

# Compile the workflow
compiled_workflow = workflow.compile()


# Add a route to test the generate node directly
@fastapi_app.post("/generate")
async def generate_route(request: QuestionRequest):
    state = {
        "question": request.question,
        "documents": [],
        "generation": "",
        "web_search": "",
    }
    try:
        outputs = []
        for output in compiled_workflow.stream(state):
            for key, value in output.items():
                outputs.append({key: value})
        return {"result": outputs}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(fastapi_app, host="0.0.0.0", port=5001)
