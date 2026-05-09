# Laboratório 09 — Arquitetura RAG Avançada (HNSW, HyDE e Cross-Encoder)

> **Disciplina:** Inteligência Artificial Aplicada  
> **Instituição:** Instituto iCEV  
> **Aluno:** Pablo Ferreira de Andrade Farias  
> **Orientador:** Prof. Dimmy  
> **Entrega:** versão `v1.0`

---

> **Nota de Integridade Acadêmica:**  
> *"Partes deste laboratório foram geradas/complementadas com IA, revisadas e validadas por Pablo Ferreira de Andrade Farias"*

> **Uso de IA:**  
> Ferramentas de IA generativa foram usadas como apoio na estruturação do pipeline, geração inicial de conteúdo técnico simulado e documentação. Todo o conteúdo foi revisado criticamente e validado pelo aluno antes da submissão.

---

## Objetivo

Este laboratório implementa um pipeline de **Retrieval-Augmented Generation (RAG)** com arquitetura de nível prático, composto por:

- busca aproximada eficiente com **HNSW**;
- transformação de consulta com **HyDE** (documento hipotético);
- refinamento de relevância com **Cross-Encoder**.

O cenário é de busca em manuais médicos técnicos a partir de queries coloquiais de paciente.

---

## Estrutura do Projeto

```text
lab9-dimmy/
├── README.md
├── requirements.txt
├── rag_lab9_step1.py
├── rag_lab9_step2.py
├── rag_lab9_step3.py
└── rag_lab9_step4.py
```

---

## Como Executar

### 1. Criar ambiente virtual e instalar dependências

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Passo 1 — Construção e indexação HNSW

```bash
python3 rag_lab9_step1.py
```

Saídas esperadas:

- `artifacts/hnsw.index`
- `artifacts/metadata.json`

### 3. Passo 2 — Query Transformation com HyDE

```bash
python3 rag_lab9_step2.py
```

Observação:

- com `OPENAI_API_KEY` configurada, o script usa chamada real ao LLM para gerar o documento hipotético técnico;
- sem `OPENAI_API_KEY`, o script usa fallback técnico local para não interromper o fluxo.

### 4. Passo 3 — Retrieval rápido (Top-10)

```bash
python3 rag_lab9_step3.py
```

Resultado esperado:

- impressão no console dos **10 documentos mais próximos** recuperados no índice HNSW usando o vetor HyDE.

### 5. Passo 4 — Re-ranking com Cross-Encoder

```bash
python3 rag_lab9_step4.py
```

Resultado esperado:

- reordenação dos 10 candidatos por score do Cross-Encoder;
- impressão dos **Top-3 finais** para injeção no contexto do LLM gerador.

---

## Explicação Técnica

### Passo 1 — Dataset, Embeddings e HNSW

Foi criado um dataset simulado com 20 fragmentos técnicos da área de saúde. Cada texto foi convertido em embedding denso com `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`. Em seguida, foi construído um índice vetorial `FAISS IndexHNSWFlat`, configurando explicitamente os hiperparâmetros de grafo.

### Passo 2 — HyDE (Hypothetical Document Embeddings)

A query coloquial do usuário é convertida em um documento técnico hipotético por LLM. Esse texto gerado é vetorizado e passa a atuar como nova âncora semântica para a etapa de recuperação.

### Passo 3 — Recuperação via Bi-Encoder

O vetor HyDE é usado para consulta por similaridade no índice HNSW, recuperando os Top-10 candidatos mais próximos (funil largo de alta cobertura).

### Passo 4 — Re-ranking com Cross-Encoder

Os 10 candidatos são reavaliados com `cross-encoder/ms-marco-MiniLM-L-6-v2`, recebendo score contextual profundo por par `[query, documento]`. Com isso, o sistema retorna os Top-3 com maior precisão semântica.

---

## Tarefa Analítica — RAM no HNSW (`M` e `ef_construction`) vs KNN exata

- `M` define o número de conexões por nó no grafo HNSW; aumentar `M` eleva consumo de RAM por armazenar mais arestas.
- `ef_construction` amplia a largura da busca durante indexação; valores maiores tendem a melhorar qualidade do grafo, com custo adicional de memória temporária e tempo de construção.
- Na KNN exata (busca exaustiva), não há grafo auxiliar, mas o custo de consulta cresce por varrer todo o conjunto. No HNSW, há overhead estrutural de memória para reduzir fortemente a latência de busca em produção.

---

## Dependências Principais

| Biblioteca | Uso no projeto |
|------------|----------------|
| `faiss-cpu` | Indexação vetorial HNSW |
| `sentence-transformers` | Embeddings e Cross-Encoder |
| `transformers` | Backend dos modelos |
| `torch` | Execução dos modelos |
| `openai` | Geração HyDE via LLM |
| `numpy` | Estruturas numéricas |

---

## Checklist de Entrega

- [x] Dataset simulado com pelo menos 20 fragmentos técnicos.
- [x] Vetorização dos textos com modelo de embedding.
- [x] Banco vetorial com índice HNSW explícito.
- [x] Explicação analítica de `M` e `ef_construction` no README.
- [x] Função HyDE para transformar query coloquial em documento técnico.
- [x] Vetorização do documento hipotético.
- [x] Recuperação Top-10 no índice HNSW.
- [x] Re-ranking com Cross-Encoder.
- [x] Impressão dos Top-3 finais.
- [x] Nota de integridade acadêmica no README.
- [ ] Tag/release final publicada como `v1.0`.

Comandos de fechamento:

```bash
git add README.md
git commit -m "docs: finaliza readme do lab 09"
git push origin main
git tag -a v1.0 -m "Entrega final Lab 09"
git push origin v1.0
```

---

## Referências

- [FAISS Documentation](https://faiss.ai/)
- [Sentence Transformers Documentation](https://www.sbert.net/)
- [HNSW (Paper)](https://arxiv.org/abs/1603.09320)
- [HyDE — Precise Zero-Shot Dense Retrieval without Relevance Labels](https://arxiv.org/abs/2212.10496)
