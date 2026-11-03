# Deliverable 2: RAG experiment

## What I built

A small RAG pipeline in `rag.py` (about 60 lines):

1. **Corpus.** `notes.txt` holds 16 short paragraphs of notes on Lipton's *The Mythos of Model Interpretability*. Each paragraph is one chunk.
2. **Embed.** `sentence-transformers/all-minilm-l6-v2` turns each chunk into a 384-dimensional vector.
3. **Index and retrieve.** FAISS (`IndexFlatIP` on normalized vectors, which gives cosine similarity) returns the 3 chunks closest to the question.
4. **Generate.** The chunks go into the prompt for `cohere/north-mini-code:free` on OpenRouter, with an instruction to answer only from that context. For comparison, the same question also goes to the model with no context.

What the libraries do: sentence-transformers handles tokenizing and running the embedding model. FAISS stores the vectors and does the nearest-neighbor search. With 16 chunks a plain numpy dot product would also work, but FAISS is what you would use at real scale.

Run it with `python rag.py "your question"`. `python test_rag.py` checks retrieval only, with no API call.

## What it did

Question: *What does it mean for a model to be simulatable?* (full output in `output.txt`)

- **Retrieval:** the top hit was the simulatability note (score 0.709). The next two hits (trust 0.475, definitions 0.426) scored well below it.
- **With context:** the model quoted Lipton's definition correctly: a person can step through every calculation from the inputs and parameters in a reasonable amount of time.
- **Without context:** the model gave a long, confident answer that was wrong for this field. It defined "simulatable" as being able to generate new data from the model.

**Takeaway:** retrieval anchored the model to the paper's meaning of a term the model misread on its own. Retrieval also makes the answer easy to explain, because you can see exactly which chunk the answer came from.

## Paper summary

<!-- TODO (Sam): one paragraph in your own words. Name the paper. Week 4 pick: Zachary C. Lipton, "The Mythos of Model Interpretability" (2016), unless the lab vote changes it. -->
