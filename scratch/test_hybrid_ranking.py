import sys, os
import re
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader
from chromadb.utils import embedding_functions
import numpy as np

reader = PdfReader('data/uploads/Nixitha_Resume (1).pdf')
pages = [f'--- Page {i+1} ---\n' + p.extract_text() for i, p in enumerate(reader.pages) if p.extract_text()]
raw = '\n\n'.join(pages)

splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, separators=['\n\n', '\n', ' ', ''])
chunks = splitter.split_text(raw)
ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name='all-MiniLM-L6-v2')

query = "explain the projects section"
q_emb = ef([query])[0]
doc_embs = ef(chunks)

# Extract meaningful keywords from query (excluding stopwords)
stopwords = {"explain", "the", "a", "an", "in", "of", "and", "or", "for", "to", "what", "is", "are", "tell", "me", "about", "my", "from"}
query_words = [w.lower() for w in re.findall(r"\w+", query) if w.lower() not in stopwords]
print("Query keywords:", query_words)

ranked = []
for i, (c, d_emb) in enumerate(zip(chunks, doc_embs)):
    # Semantic distance (lower is better, typically 0 to 2)
    dist = float(np.linalg.norm(np.array(q_emb) - np.array(d_emb)))
    
    # Keyword match score
    c_lower = c.lower()
    keyword_matches = sum(1 for kw in query_words if kw in c_lower)
    
    # Exact section header match boost (e.g. "PROJECTS", "EDUCATION")
    header_boost = sum(2 for kw in query_words if re.search(rf"\b{kw}s?\b", c_lower))
    
    # Hybrid score: lower is better; subtract boost for keyword hits
    hybrid_score = dist - (0.25 * keyword_matches) - (0.2 * header_boost)
    ranked.append({
        "chunk_idx": i,
        "dist": dist,
        "keyword_matches": keyword_matches,
        "hybrid_score": hybrid_score,
        "preview": c[:80].replace("\n", " ").encode('ascii', errors='replace').decode()
    })

ranked.sort(key=lambda x: x["hybrid_score"])
print("\n--- HYBRID RANKING RESULTS ---")
for r in ranked:
    print(f"Rank -> Chunk {r['chunk_idx']}: hybrid={r['hybrid_score']:.4f} (dist={r['dist']:.4f}, kw={r['keyword_matches']}) | {r['preview']}")
