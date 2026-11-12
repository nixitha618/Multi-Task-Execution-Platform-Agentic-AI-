import os
import pytest
from backend.app.rag.ingestion import ingestion_service
from backend.app.rag.vectorstore import rag_store

def test_document_ingestion_and_vectorstore(tmp_path):
    # Create sample text document
    doc_path = tmp_path / "sample_knowledge.txt"
    doc_path.write_text(
        "Agentic AI systems use autonomous planning and multi-agent coordination. "
        "They can dynamically invoke tools, maintain long-term memory, and self-evaluate results. "
        "ChromaDB provides scalable vector storage for semantic retrieval.",
        encoding="utf-8"
    )

    chunks = ingestion_service.load_file(str(doc_path), user_id="test_user")
    assert len(chunks) > 0
    assert "Agentic AI" in chunks[0].page_content

    # Add to vector store
    added_count = rag_store.add_documents(chunks)
    assert added_count == len(chunks)

    # Query vector store
    results = rag_store.query("What provides scalable vector storage?", top_k=2)
    assert len(results) > 0
    assert any("ChromaDB" in r["content"] for r in results)
    assert results[0]["source"] == "sample_knowledge.txt"
