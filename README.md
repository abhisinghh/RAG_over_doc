# RAG_over_doc
RAG implementation over a given document.

.item hunker is chunking a given document using langchain semantic splitter
.item Agentic RAG computes the embedding of each chunk saves them in a DataFrame, compares the embedding of the query with each chunk and retrieves rop k chunks.
.item Demo colab file is a gradio based app.



