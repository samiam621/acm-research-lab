#use FAISS to find the notes closest to the question, then send them to the model as context
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent)) #shared config.py/open_api.py live at the repo root
import faiss
from config import Config
from open_api import MODELS, get_response
from sentence_transformers import SentenceTransformer


config = Config()
EMBEDDING_MODEL_NAME = config.EMBEDDING_MODEL_NAME or "sentence-transformers/all-minilm-l6-v2"
NOTES = Path(__file__).with_name("notes.txt")
TOP_K = 3

"""so basically, you use a sentence transformer model to convert the phrases/strings into vector embeddings
and then faiss finds the closest vector points to the question."""


model = SentenceTransformer(EMBEDDING_MODEL_NAME)
chunks = [c.strip() for c in NOTES.read_text().split("\n\n") if c.strip()] #one chunk per paragraph

#normalized vectors + inner product = cosine similarity
vectors = model.encode(chunks, convert_to_numpy=True, normalize_embeddings=True)
index = faiss.IndexFlatIP(vectors.shape[1])
index.add(vectors)


def retrieve(question: str, k: int = TOP_K) -> list[tuple[float, str]]:
    q = model.encode([question], convert_to_numpy=True, normalize_embeddings=True)
    scores, ids = index.search(q, k)
    return [(float(s), chunks[i]) for s, i in zip(scores[0], ids[0])]


def ask(question: str, context: list[str] | None) -> str:
    if context:
        prompt = (
            "Answer the question using only the context below. If the context "
            "does not contain the answer, say so.\n\nContext:\n"
            + "\n".join(f"- {c}" for c in context)
            + f"\n\nQuestion: {question}"
        )
    else:
        prompt = question
    res = get_response(MODELS[1], prompt)
    return res["choices"][0]["message"]["content"]


def main():
    question = " ".join(sys.argv[1:]) or "What does it mean for a model to be simulatable?"
    hits = retrieve(question)
    print(f"Question: {question}\n\nRetrieved ({len(chunks)} chunks indexed):")
    for score, chunk in hits:
        print(f"  [{score:.3f}] {chunk}")
    print("\nAnswer with retrieved context:\n" + ask(question, [c for _, c in hits]))
    print("\nAnswer without context:\n" + ask(question, None))


if __name__ == "__main__":
    main()
