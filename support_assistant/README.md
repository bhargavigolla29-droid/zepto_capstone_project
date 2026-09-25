# Support Assistant

Eight exact policy documents from the supplied brief are stored in `docs/`. `build_index.py` loads each file, treats each document as one chunk, embeds it with `all-MiniLM-L6-v2`, and persists vectors in the `zepto_policies` ChromaDB collection.

The LangGraph graph has three required nodes: `classify_intent`, `retrieve_and_answer`, and `direct_answer`. `classify_intent` uses the specified keyword heuristic in default mock mode and conditionally routes to retrieval or direct answer. Retrieval is real in both modes and returns the top three cosine-similar chunks. Mock generation is deterministic: policy questions return `Based on the retrieved context: ...`; general questions return `I can only answer questions about Zepto policies right now.`

The final FastAPI response is validated by Pydantic as `{answer, sources, confidence}`. The prompt template in `main.py` explicitly contains role, context, task, format, length, a negative constraint, and a few-shot example for the optional real-LLM path.

Example default-mode responses after indexing:

```json
{"query":"What is the delivery fee below INR 149?"}
```

returns an answer grounded in `doc_01`, with a non-empty source list and confidence `1.0`.

```json
{"query":"Tell me a joke."}
```

returns the fixed general-question message with `sources: []` and confidence `1.0`.

## Docker

```bash
docker build -t zepto-support .
docker run --rm -p 7860:7860 zepto-support
```

## Recorded default-mode example transcripts

With `MOCK_LLM` unset/default and after `python build_index.py`, the retrieval example is expected to return the top chunk from `doc_01`:

```json
{"answer":"Based on the retrieved context: Zepto delivers grocery and household essentials to serviceable pin codes within 10 to 30 minutes of order confirmation, depending on the customer's delivery zone and current order volume. Standard delivery is free on orders over INR 149; orders below this threshold incur a flat INR 25 delivery fee.","sources":["doc_01__chunk_01"],"confidence":1.0}
```

For an unrelated query:

```json
{"answer":"I can only answer questions about Zepto policies right now.","sources":[],"confidence":1.0}
```

Feature branch validation step.
