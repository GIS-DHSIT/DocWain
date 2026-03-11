# Response Intelligence Engine — Design Document

**Goal**: Transform DocWain's retrieval and response pipeline from a patch-heavy, template-driven system into a unified intelligence layer that produces GPT-level answer quality through algorithmic response synthesis, pre-organized evidence, and closed-loop quality validation.

**Core Principle**: The intelligence lives in the planning and validation — not in the LLM. The LLM fills content; the system decides structure, format, and quality.

---

## Problem Statement

DocWain retrieves the right information but produces responses that are verbose, poorly structured, inconsistent in format, and littered with LLM artifacts. 122 iterations of regex post-processing patches have hit a ceiling. The root cause: the system asks the LLM to simultaneously understand the question, organize information, decide format, generate text, and cite sources. That's five jobs in one prompt.

The new architecture separates these concerns into distinct algorithmic stages, each with a single responsibility.

---

## Architecture Overview

### Ingestion Time (already built)

```
Document -> Extract -> Structure -> Entities/Facts -> Fingerprint -> Verify
                                         |
                                Qdrant (chunks + entity IDs)
                                Graph DB (entities + facts + relationships)
```

### Query Time (new intelligence layer)

```
User Query
    |
Conversation Graph (resolve pronouns, carry context, track disclosed facts)
    |
Query Analyzer (NLP-derived: intent geometry, entity count,
                specificity, granularity -- all algorithmic, no templates)
    |
Router Decision (graph-only vs full pipeline --
                 based on whether structured facts can fully answer)
    |
+-- Graph Path: Entity/fact lookup, sub-second response
|
+-- LLM Path:
       |
   Evidence Organizer (group by entity, deduplicate facts,
                       detect gaps, thread provenance)
       |
   Rendering Spec Generator (algorithmically derive output shape
                             from data shape + query geometry)
       |
   LLM Call (constrained by rendering spec --
             fill content, not decide format)
       |
   Output Quality Engine (validate against spec,
                          verify claims against facts,
                          restructure if non-conformant)
       |
   Clean Response
```

---

## Component 1: Conversation Graph

**Purpose**: Maintain a living understanding of the conversation across turns.

**Capabilities**:
- Track entities mentioned, facts surfaced, questions asked, answers given per session
- Resolve pronouns and references using the full conversation context, not just the last turn
- Progressive disclosure: track what has already been told to the user, surface only new information on follow-ups
- Intent continuity: carry entity context and retrieved evidence when the user shifts from factual to analytical queries
- Ambiguity resolution: rank query interpretations by weighting recent conversation topics higher

**Implementation approach**:
- Lightweight in-memory graph per session (not persisted to Neo4j)
- Nodes: entities discussed, documents referenced, facts disclosed
- Edges: "mentioned_in_turn_N", "answered_about", "compared_with"
- Builds incrementally on each turn, decays older context gracefully

---

## Component 2: Query Analyzer

**Purpose**: Derive the response shape from linguistic features algorithmically. No templates, no domain-specific branches.

**Signals analyzed**:
- Singular vs plural subject: single-entity vs multi-entity layout
- Question word geometry: "who/what" = entity-centric, "how" = process-centric, "when" = temporal ordering
- Specificity score (already computed in query_router.py): high specificity = concise answer, low = comprehensive
- Query length and complexity: short queries = direct answers, long = detailed analysis
- Entity count from NER: 1 entity = profile/card, 2+ entities = comparison/table
- Conversation context: what was already discussed narrows the scope

**Output**: A `QueryGeometry` object with intent type, expected entity count, granularity level, temporal ordering flag, comparison flag — all computed, never looked up from a table.

---

## Component 3: Evidence Organizer

**Purpose**: Restructure retrieved evidence before the LLM sees it.

**Operations**:
1. **Entity grouping**: Cluster chunks by the entities they contain (using entity IDs from Qdrant payloads). All facts about Entity A together, all facts about Entity B together.
2. **Fact deduplication**: When multiple chunks state the same fact, keep the richest version. Compare using structured FactTriples from the graph, not fuzzy text matching.
3. **Gap detection**: Compare what the query asks against what the evidence contains. Identify present fields and missing fields explicitly. Tell the LLM "no evidence found for X" to prevent hallucination.
4. **Provenance threading**: Each evidence piece carries source document name, page number, section. Preserved through to final output for natural citation.
5. **Relevance ordering**: Within each entity group, order facts by relevance to the specific query (not global relevance score).

**Key insight**: This is possible because the intel pipeline already extracted structured entities and facts at ingestion time. The organizer works on structured data, not raw text.

---

## Component 4: Rendering Spec Generator

**Purpose**: Algorithmically compute the output format specification for each query. No fixed templates.

**Three-axis analysis**:

### Axis 1: Data Shape Analysis
Examine the evidence structure:
- Ratio of KV pairs to prose to table data
- Number of distinct entities in evidence
- Field cardinality (1 value vs many values per field)
- Data density (sparse individual facts vs rich paragraphs)

### Axis 2: Query Geometry (from Component 2)
- Singular vs plural subject drives single vs multi-entity layout
- Question word drives organization: "when" = chronological, "compare" = side-by-side
- Specificity drives detail level: high = minimal, low = comprehensive

### Axis 3: Structural Inference
Composable rules that combine (not exclusive categories):
- Evidence contains 3+ entities with shared field types -> tabular layout
- Evidence is chronologically ordered -> preserve temporal sequence
- Single entity with 4+ distinct fields -> structured card layout
- Evidence is predominantly prose with no structure -> narrative
- Single fact requested -> single value, no wrapper

**Output**: A `RenderingSpec` object with: layout mode, field ordering, grouping strategy, detail level, whether to use headers/bold/tables. Generated fresh for every query.

---

## Component 5: Constrained LLM Prompter

**Purpose**: Generate LLM prompts that constrain format while freeing content.

**Approach**:
- The rendering spec is translated into structural instructions for the LLM
- The LLM is told the exact output skeleton dynamically — not from a template library, but generated from the spec
- Evidence is provided pre-organized (from Component 3), so the LLM doesn't need to sort, group, or deduplicate
- Explicit gap markers ("No evidence for X") prevent the LLM from guessing
- The LLM's job reduces to: understand the evidence, fill in the structure, write clearly

**What the LLM does NOT decide**:
- Output format (decided by rendering spec)
- Information ordering (decided by evidence organizer)
- What to include/exclude (decided by gap detection)
- How to cite sources (decided by provenance threading)

**What the LLM DOES decide**:
- Natural language phrasing
- Synthesis across multiple evidence pieces
- Inference and reasoning when explicitly required
- Completeness of explanation within each field

---

## Component 6: Output Quality Engine

**Purpose**: Closed-loop validation against the rendering specification.

**Three validation layers**:

### Structural Conformance
The rendering spec defines expected output shape. If the LLM produced prose when the spec says tabular, restructure algorithmically — parse the LLM content into semantic units and re-render in the correct format. No re-prompting needed.

### Content Integrity
Every claim in the output is traced back to evidence:
- Entity-level verification: names, organizations, dates checked against EntitySpan data
- Fact-level verification: relationships checked against FactTriple data
- Claims with no backing fact are flagged or removed
- This is structural matching on pre-computed data, not Jaccard similarity on sentences

### Cleanliness
Algorithmically derived, not regex:
- Detect LLM meta-commentary by semantic analysis: sentences that reference the act of answering rather than the answer itself are stripped
- Parse output into an AST-like structure (headings, bullets, tables, paragraphs), normalize formatting inconsistencies, re-render cleanly
- Check completeness: all entities and fields from the rendering spec should appear in the output

**Key shift**: Quality is a closed loop. Specification drives generation, output validates against specification, mismatches get corrected structurally.

---

## Graph Database Integration

### Purpose
Two roles: fast factual answers on the graph-only path, and evidence enrichment on the LLM path.

### Schema (Document-Aware)

**Three node types**:
- **Entity**: `{id, text, label, normalized, confidence, source, subscription_id, profile_id}`
- **Document**: `{id, filename, profile_id, subscription_id, fingerprint_tags}`
- **Chunk**: `{id, unit_id, qdrant_point_id, text_hash}`

**Four edge types**:
- **APPEARS_IN**: Entity -> Chunk (with char_start, char_end, confidence)
- **BELONGS_TO**: Chunk -> Document
- **RELATES_TO**: Entity -> Entity (with predicate from FactTriple, e.g., "works_at", "has_skill")
- **MENTIONED_IN**: Entity -> Document (aggregated, with frequency count)

### Profile Isolation
Every node carries `subscription_id` and `profile_id`. All Cypher queries filter on these.

### Dual Backend: Neo4j + Memgraph
A thin adapter layer speaks Cypher to whichever is available. On startup, try Neo4j first, fall back to Memgraph. Both use bolt protocol — connection URI is the only difference.

### Population
intel_pipeline.py stage 6 (Storage) writes to the graph after verification. Entities become nodes, FactTriples become RELATES_TO edges, unit_ids link to Chunk nodes, document_id links to Document nodes.

---

## Qdrant Payload Enrichment

Each chunk's Qdrant payload is enriched with:
- `entity_ids`: List of entity IDs that appear in this chunk
- `fingerprint_tags`: Auto-generated document type tags from fingerprinting
- `unit_type`: Semantic unit type (PARAGRAPH, KV_GROUP, TABLE, LIST)

Full fact details and relationships stay in the graph. Entity IDs on chunks are the bridge between the two stores — retrieve chunks from Qdrant, look up their entities in the graph for context enrichment.

---

## Query Routing

The existing `query_router.py` routes are now activated:

| Route | When | Path | Target Latency |
|-------|------|------|----------------|
| GRAPH_DIRECT | Simple fact lookup, single entity, high specificity | Graph only, no LLM | <500ms |
| HYBRID_SEARCH | Entity + context needed | Graph + Qdrant + light LLM | <2s |
| FULL_SEARCH | Broad or multi-document query | Full retrieval + LLM | <5s |
| LLM_GENERATION | Reasoning, synthesis, analysis | Full pipeline + extended LLM | <15s |

The router already computes these classifications. This design puts them to use.

---

## Integration Constraints

- No changes to existing API endpoints or request/response contracts
- Feature-flagged behind `DOCWAIN_INTEL_V2=1` environment variable
- Wraps around existing retrieval/reranking pipeline, does not replace it
- All new components are in `src/docwain_intel/` — no modifications to `src/rag_v3/` internals
- Graceful degradation: if graph DB unavailable, falls back to current pipeline behavior

---

## Success Criteria

1. Responses match the natural format a knowledgeable human would use for each query type
2. No LLM artifacts (preambles, meta-commentary, filler phrases) in any output
3. Every fact in every response traceable to source evidence
4. Simple factual queries answered in <500ms without LLM
5. Multi-turn conversations maintain context without repetition
6. System handles any document type without domain-specific code paths
