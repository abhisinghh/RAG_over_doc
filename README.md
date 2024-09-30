# RAG_over_doc
RAG implementation over a given document.

Chunker is chunking a given document using langchain semantic splitter
Agentic RAG computes the embedding of each chunk saves them in a DataFrame, compares the embedding of the query with each chunk and retrieves rop k chunks.
Demo colab file is a gradio based app.



