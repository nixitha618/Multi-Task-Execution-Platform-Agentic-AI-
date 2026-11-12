import asyncio
import os
import sys
import pandas as pd
import openpyxl

sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.abspath('.'))

from backend.app.rag.ingestion import ingestion_service
from backend.app.rag.vectorstore import rag_store
from backend.app.orchestration.graph import multi_agent_graph

async def run_all_checks():
    print('=====================================================')
    print('   COMPREHENSIVE RAG PIPELINE VERIFICATION SUITE   ')
    print('=====================================================')

    # --- Check 1: Multi-Page PDF Ingestion & Chunking ---
    print('\n[CHECK 1] Ingesting PratishaSummerInternhsip.pdf...')
    pdf_path = os.path.join('data', 'uploads', 'PratishaSummerInternhsip.pdf')
    assert os.path.exists(pdf_path), 'PDF file must exist in data/uploads/'
    pdf_chunks = ingestion_service.load_file(pdf_path, user_id='test_rag_user')
    assert len(pdf_chunks) > 0, 'PDF chunks should be non-empty'
    pages = {c.metadata.get('page') for c in pdf_chunks}
    print(f'-> Total Chunks: {len(pdf_chunks)}, Pages Tracked: {len(pages)} (min: {min(pages)}, max: {max(pages)})')
    assert len(pages) == 22, f'Expected 22 pages tracked, found {len(pages)}'
    
    res_add1 = rag_store.add_documents(pdf_chunks, force_reload=True)
    assert res_add1.get('chunks_added') == len(pdf_chunks)
    print(f"-> Indexed {res_add1.get('chunks_added')} chunks into ChromaDB.")

    # --- Check 2: Hash-Based Duplicate Prevention ---
    print('\n[CHECK 2] Testing SHA-256 Duplicate Detection...')
    res_add2 = rag_store.add_documents(pdf_chunks, force_reload=False)
    print(f"-> Ingestion Result: already_indexed={res_add2.get('already_indexed')}, message={res_add2.get('message')}")
    assert res_add2.get('already_indexed') is True, 'Duplicate file must be detected and rejected'

    # --- Check 3: Multi-Format Ingestion (CSV, XHLX, TXT) ---
    print('\n[CHECK 3] Testing Multi-Format Ingestion (CSV, XHLX, TXT)...')
    test_dir = os.path.join('data', 'test_rag_docs')
    os.makedirs(test_dir, exist_ok=True)

    # CSV
    csv_path = os.path.join(test_dir, 'sample_employees.csv')
    df_csv = pd.DataFrame([
        {'id': 101, 'name': 'Alice Johnson', 'role': 'Lead Architect', 'department': 'Agentic AI Systems'},
        {'id': 102, 'name': 'Bob Smith', 'role': 'MLOps Engineer', 'department': 'Vector Search & Embeddings'}
    ])
    df_csv.to_csv(csv_path, index=False)
    csv_chunks = ingestion_service.load_file(csv_path, user_id='test_rag_user')
    assert len(csv_chunks) >= 1
    rag_store.add_documents(csv_chunks)
    print(f'-> CSV Ingestion: {len(csv_chunks)} chunks parsed.')

    # XLSX
    xlsx_path = os.path.join(test_dir, 'sample_projects.xlsx')
    df_xlsx = pd.DataFrame([
        {'Project': 'Project Phoenix', 'Budget': 150000, 'Status': 'Active', 'Lead': 'Dr. Chen'},
        {'Project': 'Project Apex', 'Budget': 280000, 'Status': 'Completed', 'Lead': 'Sarah Connor'}
    ])
    df_xlsx.to_excel(xlsx_path, index=False)
    xlsx_chunks = ingestion_service.load_file(xlsx_path, user_id='test_rag_user')
    assert len(xlsx_chunks) >= 1
    rag_store.add_documents(xlsx_chunks)
    print(f'-> XLSX Ingestion: {len(xlsx_chunks)} chunks parsed.')

    # TXT
    txt_path = os.path.join(test_dir, 'sample_notes.txt')
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write('The quarterly technology review meeting is scheduled for October 15th in Conference Room B.')
    txt_chunks = ingestion_service.load_file(txt_path, user_id='test_rag_user')
    assert len(txt_chunks) == 1
    rag_store.add_documents(txt_chunks)
    print(f'-> TXT Ingestion: {len(txt_chunks)} chunk parsed.')

    # --- Check 4: Semantic Query Retrieval & Citations ---
    print('\n[CHECK 4] Testing Semantic Retrieval with Similarity Scores...')
    q_matches = rag_store.retrieve_relevant_documents(
        query='who is the candidate for the summer internship certificate at meeloop',
        top_k=4,
        source='PratishaSummerInternhsip.pdf'
    )
    assert len(q_matches) > 0
    top = q_matches[0]
    print(f'-> Top Match: Score={top["score"]}, Page={top["page"]}, Source={top["source"]}')
    assert top['source'] == 'PratishaSummerInternhsip.pdf'
    assert top['score'] > 0.45, f'Score should be robust, got {top["score"]}'

    # --- Check 5: Document Deletion by ID ---
    print('\n[CHECK 5] Testing Document Deletion by document_id...')
    csv_doc_id = csv_chunks[0].metadata.get('document_id')
    before_count = rag_store.count()
    deleted = rag_store.delete_document(csv_doc_id)
    after_count = rag_store.count()
    print(f'-> Deleted {deleted} chunks for document_id={csv_doc_id} (Count: {before_count} -> {after_count})')
    assert deleted == len(csv_chunks)
    assert after_count == before_count - deleted

    # --- Check 6: Refusal on Absent Context ---
    print('\n[CHECK 6] Testing Refusal When Context is Absent...')
    unrelated_matches = rag_store.retrieve_relevant_documents(
        query='quantum gravitational field equations in 11 dimensional string theory',
        top_k=3,
        threshold=0.85
    )
    print(f'-> Unrelated matches above 0.85 threshold: {len(unrelated_matches)}')
    assert len(unrelated_matches) == 0, 'Should filter out unrelated content below threshold'

    # --- Check 7: End-to-End Multi-Agent Execution ---
    print('\n[CHECK 7] Testing End-to-End Multi-Agent Workflow on Pratisha PDF...')
    initial_state = {
        'task_id': 'test_e2e_rag_suite',
        'user_id': 'default_user',
        'user_prompt': 'what is internship certificate',
        'attached_files': [os.path.abspath(pdf_path)],
        'chat_history': [],
        'memory_context': '',
        'execution_plan': {},
        'step_results': {},
        'artifacts': [],
        'events': [],
        'retry_count': 0,
        'max_retries': 2,
        'needs_retry': False,
        'retry_instructions': None,
        'final_output': None,
        'status': 'PENDING'
    }

    result = await multi_agent_graph.ainvoke(initial_state)
    assert result.get('status') == 'COMPLETED'
    final_text = result.get('final_response') or result.get('final_output', '')
    assert 'certificate' in final_text.lower(), 'Must discuss certificate'
    assert 'internship' in final_text.lower(), 'Must discuss internship'
    assert 'pratishasummerinternhsip.pdf' in final_text.lower(), 'Must cite source document'
    print('\n[PASS] All 7 Comprehensive RAG Checks Passed Perfectly!')

if __name__ == '__main__':
    asyncio.run(run_all_checks())
