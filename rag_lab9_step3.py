import json
import os
from pathlib import Path

import faiss
from sentence_transformers import SentenceTransformer

from rag_lab9_step2 import DEFAULT_QUERY, EMBED_MODEL_NAME, generate_hypothetical_document, vectorize_text

OUTPUT_DIR = Path("artifacts")
INDEX_PATH = OUTPUT_DIR / "hnsw.index"
META_PATH = OUTPUT_DIR / "metadata.json"


def load_index_and_metadata() -> tuple[faiss.Index, dict]:
    if not INDEX_PATH.exists() or not META_PATH.exists():
        raise FileNotFoundError(
            "Arquivos do passo 1 não encontrados. Rode 'python rag_lab9_step1.py' primeiro."
        )

    index = faiss.read_index(str(INDEX_PATH))
    metadata = json.loads(META_PATH.read_text(encoding="utf-8"))
    return index, metadata


def retrieve_top_k(index: faiss.Index, query_vector, k: int = 10):
    distances, indices = index.search(query_vector, k)
    return distances[0], indices[0]


def main() -> None:
    query = os.getenv("QUERY", DEFAULT_QUERY)

    hypothetical_doc = generate_hypothetical_document(query)
    embed_model = SentenceTransformer(EMBED_MODEL_NAME)
    query_vector = vectorize_text(hypothetical_doc, embed_model=embed_model)

    index, metadata = load_index_and_metadata()
    scores, doc_ids = retrieve_top_k(index, query_vector, k=10)

    print("Passo 3 concluído:")
    print(f"- Query original: {query}")
    print(f"- Documento HyDE: {hypothetical_doc}")
    print("- Top-10 documentos recuperados (HNSW + similaridade):")

    docs = metadata["documents"]
    for rank, (doc_id, score) in enumerate(zip(doc_ids, scores), start=1):
        if doc_id < 0:
            continue
        doc_text = docs[doc_id]["text"]
        print(f"{rank:02d}. id={doc_id} | score={score:.4f} | {doc_text}")


if __name__ == "__main__":
    main()
