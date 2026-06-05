# RAG Learning Project

This repo builds a production-style RAG system step by step, starting with data ingestion and a minimal baseline.

## Phase 0 Goals
- Curate a small public insurance corpus (50-100 docs)
- Define a document/chunk schema with traceable citations
- Set up a clean data layout

## Data Layout
- data/raw: original PDFs or HTML
- data/clean: normalized text
- data/artifacts: derived outputs (chunks, embeddings later)
- data/lists: corpus tracking lists

## Next Step
Populate the corpus list in docs/corpus_list.md with 20-30 public insurance documents.

## Phase 1: Retrieval Baseline
1. Build embeddings + FAISS index from chunks.
2. Query the index to validate retrieval quality.

## Phase 2: Hybrid Retrieval
1. Build BM25 index from chunks.
2. Query with hybrid BM25 + vector scoring.

## Phase 3: Reranking
1. Rerank hybrid candidates with a cross-encoder.

## Phase 1: Minimal RAG Answering
1. Use OpenAI to answer with citations from retrieved chunks.
2. Use Ollama to answer offline with citations from retrieved chunks.
