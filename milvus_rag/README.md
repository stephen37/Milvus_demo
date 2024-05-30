# Local RAG application with Milvus
This Jupyter notebook is designed to showcase how to build a RAG (Retrieval Augmented Generations) app using [Milvus-Lite](https://github.com/milvus-io/milvus-lite), Langchain and Ollama.

It is designed to run locally and to show that you can build a RAG app in less than 50 lines of code! With Milvus Lite, you don't need to use Docker or anything. It will just run in your notebook

## Requirements
- [Ollama](https://ollama.com/) -- Run LLMs locally. Make sure you have the model downloaded as well, e.g. in our case we use Llama3 so run `ollama run llama3`. 

## Installation
Use either Poetry or pip to install packages.

- If you already have poetry installed: `poetry install`
- If you don't, you can use `pip install -r requirements.txt` 

Then start your jupyter notebook and exectute the different cells.

---

Feel free to check out [Milvus](https://github.com/milvus-io/milvus), and share your experiences with the community by joining our [Discord](https://discord.gg/FG6hMJStWu).

