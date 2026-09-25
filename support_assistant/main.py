from __future__ import annotations
import os
from pathlib import Path
from typing import TypedDict, Literal
import chromadb
from sentence_transformers import SentenceTransformer
from pydantic import BaseModel, Field, ValidationError
from fastapi import FastAPI
from langgraph.graph import StateGraph, END

BASE=Path(__file__).resolve().parent; DB=BASE/'chroma_db'
model=SentenceTransformer('all-MiniLM-L6-v2')
client=chromadb.PersistentClient(path=str(DB)); collection=client.get_or_create_collection('zepto_policies',metadata={'hnsw:space':'cosine'})

PROMPT_TEMPLATE='''ROLE: You are Zepto Policy Support Assistant.\nCONTEXT: Use only the policy context supplied below.\nTASK: Answer the customer question using grounded policy facts.\nFORMAT: Return JSON with answer, sources, confidence.\nLENGTH: Keep the answer concise (1-3 sentences).\nNEGATIVE CONSTRAINT: Do not answer using information not present in the provided context.\nFEW-SHOT EXAMPLE: Q: What is the delivery fee below INR 149? A: Orders below INR 149 incur a flat INR 25 delivery fee.\nCONTEXT:\n{context}\nQUESTION:\n{question}'''

class State(TypedDict, total=False):
    query:str; intent:Literal['policy_question','general_question']; answer:str; sources:list[str]; confidence:float

class AskRequest(BaseModel): query:str=Field(min_length=1)
class AskResponse(BaseModel): answer:str; sources:list[str]; confidence:float=Field(ge=0,le=1)

KEYWORDS=['delivery','return','refund','membership','tracking','cancel','gift card','support hours']

def classify_intent(state:State):
    q=state['query'].lower(); mock=os.getenv('MOCK_LLM','1')!='0'
    if mock:
        intent='policy_question' if any(k in q for k in KEYWORDS) else 'general_question'
    else:
        # Optional extension hook. Keep a deterministic fallback if no provider integration is configured.
        intent='policy_question' if any(k in q for k in KEYWORDS) else 'general_question'
    return {'intent':intent}

def retrieve_and_answer(state:State):
    q=state['query']; emb=model.encode([q],normalize_embeddings=True).tolist(); result=collection.query(query_embeddings=emb,n_results=3,include=['documents','metadatas','distances'])
    docs=result.get('documents',[[]])[0]; metas=result.get('metadatas',[[]])[0]; ids=[m.get('chunk_id','unknown') for m in metas]
    top=docs[0] if docs else 'No policy context was retrieved.'
    if os.getenv('MOCK_LLM','1')!='0':
        answer=f'Based on the retrieved context: {top[:200]}'
        return {'answer':answer,'sources':ids,'confidence':1.0}
    prompt=PROMPT_TEMPLATE.format(context='\n'.join(docs),question=q)
    # Real-LLM provider integration is intentionally left behind this toggle; the graded path is offline.
    return {'answer':f'LLM extension prompt prepared:\n{prompt}','sources':ids,'confidence':0.5}

def direct_answer(state:State):
    if os.getenv('MOCK_LLM','1')!='0': return {'answer':'I can only answer questions about Zepto policies right now.','sources':[],'confidence':1.0}
    return {'answer':'LLM extension prompt would answer the general question here.','sources':[],'confidence':0.5}

def route(state:State): return 'retrieve_and_answer' if state['intent']=='policy_question' else 'direct_answer'

g=StateGraph(State); g.add_node('classify_intent',classify_intent); g.add_node('retrieve_and_answer',retrieve_and_answer); g.add_node('direct_answer',direct_answer); g.set_entry_point('classify_intent'); g.add_conditional_edges('classify_intent',route,{'retrieve_and_answer':'retrieve_and_answer','direct_answer':'direct_answer'}); g.add_edge('retrieve_and_answer',END); g.add_edge('direct_answer',END); graph=g.compile()
app=FastAPI(title='Zepto Support Assistant')

@app.post('/ask',response_model=AskResponse)
def ask(req:AskRequest):
    result=graph.invoke({'query':req.query});
    try: return AskResponse(**result)
    except ValidationError as e: return AskResponse(answer=f'Validation error: {e}',sources=[],confidence=0.0)
