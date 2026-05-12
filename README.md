Enterprise AI Chatbot with RAG & Local LLM
A self-hosted, Retrieval-Augmented Generation (RAG) chatbot system designed for automated internal knowledge retrieval. This project integrates a local Large Language Model (LLM) with a vector database to provide precise, context-aware responses based on private documentation.

Key Features
- Semantic Retrieval (RAG): Leverages PostgreSQL with pgvector to perform high-accuracy similarity searches on embedded documentation.
- Local LLM Integration: Powered by Ollama, ensuring 100% data privacy by keeping all computations and data within the local infrastructure.
- Conversational Memory: Implements persistent chat history using LangChain’s PostgresChatMessageHistory, allowing for multi-turn dialogues.
- Hybrid Search Logic: Combines semantic similarity with keyword-based ranking to enhance the relevance of retrieved context.

Tech Stack
- Language: Python (Flask)
- AI/ML: Ollama, Sentence-Transformers
- Orchestration: LangChain
- Database: PostgreSQL + pgvector
- Environment: Self-hosted / Local AI Infrastructure

Performance & Evaluation
- Accuracy: Achieved an average Contextual Similarity Score of 70% during internal validation, measuring the alignment between AI responses and ground-truth documentation.
- Efficiency: Streamlined information retrieval by automating the search across thousands of product and service documents.

System Architecture

The chatbot follows a structured RAG pipeline:
- Document Ingestion: PDF/Text documents are split into optimized chunks.
- Embedding: Chunks are transformed into 384-dimensional vectors using all-MiniLM-L6-v2.
- Storage: Vectors are stored in a PostgreSQL database equipped with the pgvector extension.
- Inference: Upon user query, the system retrieves the most relevant context and generates a response via the local Ollama API.

Disclaimer

    Note: This repository is a sanitized version of the production implementation. Proprietary company data and sensitive credentials have been replaced with generic templates to comply with data privacy policies.
