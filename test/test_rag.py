import sys
import os

project_path = r"D:\PSX - Project"
if project_path not in sys.path:
    sys.path.append(project_path)

# ✅ Import the NEW multi-query function
from agents_graph import fetch_multi_query_rag
from ingestion_report import ingest_quarterly_pdf

def test_multi_query():
    target_symbol = "KEL"
    
    print(f"🔍 Testing Multi-Query Retrieval for {target_symbol}...")
    
    # This function searches for P&L, Balance Sheet, AND Cash Flow simultaneously
    docs = fetch_multi_query_rag(target_symbol)
    
    if not docs:
        print("❌ No chunks returned. You may need to re-ingest.")
        return
        
    print(f"✅ Found {len(docs)} targeted chunks (Expected: 3).")
    
    print("\n" + "="*70)
    print(f"📄 EXACT PAYLOAD BEING SENT TO THE LLM FOR {target_symbol}")
    print("="*70)
    
    for i, text_content in enumerate(docs):
        print(f"\n--- CHUNK {i+1} ---")
        # Print first 1000 characters so we can see the headers
        print(text_content[:1000] + ("..." if len(text_content) > 1000 else ""))
        print("-" * 70)

if __name__ == "__main__":
    test_multi_query()