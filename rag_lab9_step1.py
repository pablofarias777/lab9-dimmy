import json
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

DATASET = [
    "Paciente com cefaleia pulsátil hemicraniana, fotofobia e fonofobia, com aura visual escotomática prévia.",
    "Quadro de dispneia paroxística noturna associado a estertores bibasais e edema periférico compatível com congestão.",
    "Dor retroesternal em aperto irradiando para membro superior esquerdo, sudorese fria e náusea intensa.",
    "Poliúria, polidipsia e perda ponderal, com glicemia capilar persistentemente elevada e cetonúria discreta.",
    "Episódios de sibilância expiratória difusa com resposta parcial a broncodilatador beta-2 de curta duração.",
    "Lombalgia mecânica crônica sem sinais de radiculopatia, piora à flexoextensão e alívio em repouso relativo.",
    "Diarreia aquosa de início agudo sem sangue, com dor abdominal tipo cólica e desidratação leve.",
    "Exantema maculopapular pruriginoso pós-exposição medicamentosa, sem comprometimento de mucosas.",
    "Quadro de ansiedade generalizada com hiperatividade autonômica, insônia de manutenção e ruminação cognitiva.",
    "Otalgia unilateral com hiperemia de membrana timpânica e redução transitória da acuidade auditiva.",
    "Disúria terminal e urgência miccional com piúria em exame tipo I, sugestivo de infecção urinária baixa.",
    "Dor epigástrica em queimação pós-prandial tardia com melhora parcial após inibidor de bomba de prótons.",
    "Febre alta intermitente, mialgia intensa e plaquetopenia, com sorologia reagente para arbovirose.",
    "Rigidez matinal poliarticular superior a uma hora, edema de pequenas articulações e fator reumatoide positivo.",
    "Síncope vasovagal precedida de pródromos autonômicos, sem evidências de arritmia em monitorização inicial.",
    "Dermatite atópica em flexuras com liquenificação e prurido noturno de difícil controle clínico.",
    "Hematoquezia intermitente em paciente acima de 50 anos, demandando rastreio endoscópico de neoplasia.",
    "Rebaixamento do nível de consciência em contexto de hiponatremia grave e osmolaridade plasmática reduzida.",
    "Tosse seca persistente há oito semanas com radiografia sem infiltrado, hipótese de síndrome de gotejamento pós-nasal.",
    "Parestesia distal em padrão luva e bota, associada a neuropatia periférica de provável etiologia metabólica.",
]


OUTPUT_DIR = Path("artifacts")
INDEX_PATH = OUTPUT_DIR / "hnsw.index"
META_PATH = OUTPUT_DIR / "metadata.json"


def build_embeddings(texts: list[str]) -> np.ndarray:
    model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    vectors = model.encode(texts, normalize_embeddings=True)
    return np.asarray(vectors, dtype="float32")


def build_hnsw_index(vectors: np.ndarray, m: int = 32, ef_construction: int = 200) -> faiss.IndexHNSWFlat:
    dim = vectors.shape[1]
    index = faiss.IndexHNSWFlat(dim, m)
    index.hnsw.efConstruction = ef_construction
    index.metric_type = faiss.METRIC_INNER_PRODUCT
    index.add(vectors)
    return index


def save_artifacts(index: faiss.IndexHNSWFlat, texts: list[str], m: int, ef_construction: int) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(INDEX_PATH))

    metadata = {
        "total_documents": len(texts),
        "m": m,
        "ef_construction": ef_construction,
        "documents": [{"id": i, "text": txt} for i, txt in enumerate(texts)],
    }
    META_PATH.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    m = 32
    ef_construction = 200

    vectors = build_embeddings(DATASET)
    index = build_hnsw_index(vectors, m=m, ef_construction=ef_construction)
    save_artifacts(index, DATASET, m=m, ef_construction=ef_construction)

    print("Passo 1 concluído:")
    print(f"- Documentos indexados: {len(DATASET)}")
    print(f"- Dimensão dos vetores: {vectors.shape[1]}")
    print(f"- Índice salvo em: {INDEX_PATH}")
    print(f"- Metadados salvos em: {META_PATH}")


if __name__ == "__main__":
    main()
