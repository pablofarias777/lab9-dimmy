# Laboratório 09 - Arquitetura RAG Avançada



## Passo 1 (concluído)

- Base simulada com 20 fragmentos técnicos da área de saúde.
- Embeddings gerados com `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`.
- Indexação vetorial com HNSW via FAISS (`IndexHNSWFlat`).

### Como executar

```bash
pip install -r requirements.txt
python rag_lab9_step1.py
```

### Saídas

- `artifacts/hnsw.index`: índice HNSW serializado.
- `artifacts/metadata.json`: metadados e documentos indexados.

## Tarefa analítica: impacto de `M` e `ef_construction` no uso de RAM

- `M` controla quantas conexões cada nó do grafo mantém. Quando `M` aumenta, cresce o número de arestas por vetor, então o índice consome mais memória RAM.
- `ef_construction` controla a largura de busca durante a construção do índice. Valores maiores aumentam custo temporário de memória e tempo de indexação, mas melhoram a qualidade dos vizinhos inseridos.
- Em comparação com KNN exata (varredura de todos os vetores), HNSW troca um pouco de memória extra do grafo por latência muito menor em consulta, mantendo boa precisão aproximada.

## Declaração obrigatória

"Partes deste laboratório foram geradas/complementadas com IA, revisadas e validadas por Pablo Farias"
