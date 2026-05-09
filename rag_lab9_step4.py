import json
import os
from pathlib import Path

import faiss
from sentence_transformers import CrossEncoder, SentenceTransformer

from rag_lab9_step2 import DEFAULT_QUERY, EMBED_MODEL_NAME, generate_hypothetical_document, vectorize_text

OUTPUT_DIR = Path("artifacts")
INDEX_PATH = OUTPUT_DIR / "hnsw.index"
META_PATH = OUTPUT_DIR / "metadata.json"
CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


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


def rerank_with_cross_encoder(query: str, candidates: list[dict]) -> list[dict]:
    model = CrossEncoder(CROSS_ENCODER_MODEL)
    pairs = [[query, item["text"]] for item in candidates]
    ce_scores = model.predict(pairs)

    reranked = []
    for item, ce_score in zip(candidates, ce_scores):
        row = item.copy()
        row["cross_score"] = float(ce_score)
        reranked.append(row)

    reranked.sort(key=lambda x: x["cross_score"], reverse=True)
    return reranked


def main() -> None:
    query = os.getenv("QUERY", DEFAULT_QUERY)

    hypothetical_doc = generate_hypothetical_document(query)
    embed_model = SentenceTransformer(EMBED_MODEL_NAME)
    query_vector = vectorize_text(hypothetical_doc, embed_model=embed_model)

    index, metadata = load_index_and_metadata()
    bi_scores, doc_ids = retrieve_top_k(index, query_vector, k=10)

    docs = metadata["documents"]
    candidates = []
    for doc_id, bi_score in zip(doc_ids, bi_scores):
        if doc_id < 0:
            continue
        candidates.append(
            {
                "id": int(doc_id),
                "bi_score": float(bi_score),
                "text": docs[doc_id]["text"],
            }
        )

    reranked = rerank_with_cross_encoder(query, candidates)
    top3 = reranked[:3]

    print("Passo 4 concluído:")
    print(f"- Query original: {query}")
    print("- Top-10 do bi-encoder reranqueados por Cross-Encoder:")
    for rank, item in enumerate(reranked, start=1):
        print(
            f"{rank:02d}. id={item['id']} | bi_score={item['bi_score']:.4f} "
            f"| cross_score={item['cross_score']:.4f} | {item['text']}"
        )

    print("- Top-3 finais para contexto do LLM:")
    for rank, item in enumerate(top3, start=1):
        print(f"{rank}. id={item['id']} | cross_score={item['cross_score']:.4f} | {item['text']}")


if __name__ == "__main__":
    main()
