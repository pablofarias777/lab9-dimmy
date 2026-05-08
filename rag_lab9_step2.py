import os
from typing import Optional

import numpy as np
from openai import OpenAI
from sentence_transformers import SentenceTransformer

EMBED_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
DEFAULT_QUERY = "dor de cabeça latejante e luz incomodando"


def generate_hypothetical_document(query: str, llm_model: str = "gpt-4.1-mini") -> str:
    """Gera um documento técnico hipotético (HyDE) a partir de query coloquial."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return (
            "Paciente com cefaleia pulsátil, provável padrão enxaquecoso, associada a fotofobia "
            "e possível fonofobia, sem sinais imediatos de déficit neurológico focal. "
            f"Descrição original do paciente: {query}."
        )

    client = OpenAI(api_key=api_key)
    prompt = (
        "Você é um assistente médico técnico. Converta a queixa coloquial do paciente em um "
        "parágrafo de prontuário com jargão clínico, sem diagnóstico definitivo e sem inventar "
        "dados críticos ausentes.\n"
        f"Queixa do paciente: {query}\n"
        "Saída: um único parágrafo técnico em português."
    )

    response = client.responses.create(
        model=llm_model,
        input=prompt,
        temperature=0.2,
        max_output_tokens=220,
    )
    return response.output_text.strip()


def vectorize_text(text: str, embed_model: Optional[SentenceTransformer] = None) -> np.ndarray:
    model = embed_model or SentenceTransformer(EMBED_MODEL_NAME)
    vector = model.encode([text], normalize_embeddings=True)
    return np.asarray(vector, dtype="float32")


def main() -> None:
    query = os.getenv("QUERY", DEFAULT_QUERY)

    hypothetical_doc = generate_hypothetical_document(query)
    embed_model = SentenceTransformer(EMBED_MODEL_NAME)
    query_vector = vectorize_text(hypothetical_doc, embed_model=embed_model)

    print("Passo 2 concluído:")
    print(f"- Query original: {query}")
    print(f"- Documento hipotético (HyDE): {hypothetical_doc}")
    print(f"- Shape do vetor HyDE: {query_vector.shape}")


if __name__ == "__main__":
    main()
