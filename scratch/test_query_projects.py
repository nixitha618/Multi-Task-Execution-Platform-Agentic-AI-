import sys, os
sys.path.insert(0, os.path.abspath("."))
from backend.app.rag.vectorstore import rag_store

res = rag_store.query('explain the projects section', top_k=8, source='Nixitha_Resume (1).pdf')
print('Matches count:', len(res))
for i, r in enumerate(res):
    print(f"Match {i}: chunk {r['page']}, dist={r['distance']}")
    print(r['content'][:150].encode('ascii', errors='replace').decode())
