# TODO

## Frontend
- [ ] Replace placeholder assistant response with real RAG answer.
- [ ] Remove "boilerplate" wording in UI text.
- [ ] Wire prompt to retriever + generator.

## Retrieval and Verification
- [ ] Render retrieved chunks in context expander.
- [ ] Show source metadata (doc, section/page, chunk id).
- [ ] Show retrieval scores/ranking.
- [ ] Add answer provenance summary.

## Controls
- [ ] Wire collection -> index/namespace.
- [ ] Wire top_k -> retrieval count.
- [ ] Wire temperature -> generation setting.
- [ ] Keep show_context toggle behavior.

## Reliability
- [ ] Add retrieval/generation error handling.
- [ ] Add loading/status indicator.
- [ ] Add optional chat memory mode.
- [ ] Add reset settings button.

## Models and Embeddings
- [ ] Primary: qwen2.5:7b-instruct + bge-small-en-v1.5.
- [ ] Alternative 2: llama3.1:8b-instruct + bge-base-en-v1.5.
- [ ] Alternative 3: mistral:7b-instruct + nomic-embed-text.

## Project Structure
- [ ] Split retrieval, generation, and data models into modules.
- [ ] Add config defaults for model/index.
- [ ] backend requirementxtxt
