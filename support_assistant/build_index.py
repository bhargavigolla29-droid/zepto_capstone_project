from pathlib import Path
import chromadb
from sentence_transformers import SentenceTransformer
BASE=Path(__file__).resolve().parent
DB=BASE/'chroma_db'
DOCS=BASE/'docs'
COLLECTION='zepto_policies'

def build():
    client=chromadb.PersistentClient(path=str(DB)); col=client.get_or_create_collection(COLLECTION,metadata={'hnsw:space':'cosine'})
    model=SentenceTransformer('all-MiniLM-L6-v2')
    ids=[]; docs=[]; metas=[]
    for p in sorted(DOCS.glob('doc_*.txt')):
        text=p.read_text(encoding='utf-8').strip(); ids.append(p.stem+'__chunk_01'); docs.append(text); metas.append({'document_id':p.stem,'chunk_id':p.stem+'__chunk_01'})
    embeddings=model.encode(docs,normalize_embeddings=True).tolist()
    if ids:
        try: col.upsert(ids=ids,documents=docs,embeddings=embeddings,metadatas=metas)
        except Exception: col.add(ids=ids,documents=docs,embeddings=embeddings,metadatas=metas)
    print(f'Indexed {len(ids)} chunks into {COLLECTION}')
if __name__=='__main__': build()
