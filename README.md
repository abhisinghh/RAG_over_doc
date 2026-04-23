# RAG_over_doc
RAG implementation over a given document.
This Repo is a Work in Progress for a production-grade RAG system.
Aim and Objective:
RAG  for document intelligence, designed to explore and compare dense, graph-based, and agentic retrieval architectures. This project focuses on building reliable, high-performance LLM pipelines with emphasis on retrieval quality, reranking, and hallucination reduction.

It implements end-to-end workflows including embedding-based retrieval, graph traversal using structured relationships, and agent-driven reasoning for complex queries. The system is built with a strong focus on evaluation, modular design, and scalability, reflecting real-world production constraints rather than isolated experimentation.

Designed to bridge research ideas with practical deployment, this repository demonstrates how different RAG paradigms behave across accuracy, faithfulness, and system performance.
##
- chunker.py 1: Chunker is chunking a given document using langchain semantic splitter
- Agentic RAG.py:Agentic RAG computes the embedding of each chunk saves them in a DataFrame, compares the embedding of the query with each chunk and retrieves rop k chunks.
- dense_rag_demo.ipynb "  Demo collab file is a gradio based app.
##


