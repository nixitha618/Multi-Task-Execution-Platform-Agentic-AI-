import sys, os
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
q_emb = ef(['explain the projects section'])[0]
doc_embs = ef(chunks)

for i, d_emb in enumerate(doc_embs):
    dist = np.linalg.norm(np.array(q_emb) - np.array(d_emb))
    clean_preview = chunks[i][:60].encode('ascii', errors='replace').decode()
    print(f'Chunk {i}: dist={dist:.4f}, preview: {clean_preview}...')
