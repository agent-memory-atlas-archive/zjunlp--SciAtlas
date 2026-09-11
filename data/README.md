# SciAtlasEval

SciAtlasEval is an evaluation suite for latent-relation recovery, scientific-object
retrieval and research-idea assessment. It contains 200 HQR cases, 70 FCR queries
and 349 Idea entries. Data are stored as UTF-8 JSON Lines, one record per line.

## HQR — Heterogeneous Query–Object Retrieval

HQR evaluates retrieval of papers and authors from paper, concept and author
queries after direct revealing relations are masked. Concepts are represented
as keyword nodes. The six subtasks contain 30 paper-to-paper, 45 paper-to-author,
35 concept-to-paper, 30 concept-to-author, 40 author-to-paper and 20
author-to-author cases. Each case has 5–20 gold targets.

| File | Contents |
| --- | --- |
| `HQR/queries.jsonl` | 168 queries with text, object type and source-node identifiers. |
| `HQR/cases.jsonl` | 200 cases specifying each query's subtask and target type. |
| `HQR/gold.jsonl` | Gold targets and supporting evidence for each case. |
| `HQR/masks.jsonl` | Relations to mask separately for each case. |

Use `query_id` to link queries to cases and `case_id` to link cases, gold targets
and masks. A query may be used in multiple cases. Methods return up to 20
targets and are evaluated using Recall@20, nDCG@20 and MRR.

## FCR — Future Context Recovery

FCR evaluates whether historical literature can recover future scientific
context. The 70 queries represent historical papers using their titles and
abstracts. Retrieval is restricted to papers published in or before 2022.
Retrieved papers are projected through citation links to papers published in
2023–2025, excluding the query paper itself as a projection source.

| File | Contents |
| --- | --- |
| `FCR/queries.jsonl` | Historical-paper queries, source identifiers and publication years. |
| `FCR/gold.jsonl` | Future-paper targets and the recoverable gold sets used for evaluation. |
| `FCR/projection_rules.jsonl` | Projection-source exclusions for each query. |

Link records by `query_id`. Methods retrieve up to 100 historical papers;
evaluation uses `recoverable_gold_targets` with Recall@20, nDCG@20 and MRR.
The graph and citation data required to run retrieval and projection are
separate from the query and annotation files in this package.

## Idea — Research-Idea Retrieval and Assessment

Idea contains 349 paper-grounded entries for idea-level retrieval,
contextualization and innovation assessment.

| File | Contents |
| --- | --- |
| `Idea/ideas.jsonl` | Structured ideas, combined idea text and parsed source-paper text. |
| `Idea/source_papers.jsonl` | Source titles, abstracts, domain labels and available identifiers. |

Link the two files using `source_paper_id`; `idea_id` identifies each idea entry.
The structured fields are:

- `basic_idea`: a concise overview of the research idea.
- `motivation`: the research problem and the gap addressed.
- `method`: the principal technical, empirical or conceptual approach.
- `experimental_focus`: the experimental objectives and validation focus,
  corresponding to the experiment component described in the paper.
- `impact`: the intended contribution or potential significance of the work.

The first four fields are arrays of strings; `impact` is a paragraph.
`idea_text` combines the components, including impact. `source_full_text`
contains the parsed source text associated with the entry.
