# DocWain Knowledge Graph Schema

This schema is intentionally minimal and generic so it can represent resumes, invoices, legal documents, tax forms, and other document types without hardcoded rules.

## Nodes

### `Document`
**Properties:**
- `doc_id` (string, unique)
- `profile_id` (string)
- `subscription_id` (string)
- `doc_name` (string)
- `document_category` (string)
- `detected_language` (string)
- `created_at` (datetime)
- `graph_version` (string)

### `Entity`
**Properties:**
- `entity_id` (string, unique; `${type}::${normalized_name}`)
- `name` (string)
- `type` (string, e.g., PERSON, ORGANIZATION, SKILL, LOCATION, DATE, AMOUNT, CLAUSE, TERM)
- `normalized_name` (string)

**Optional labels** (applied when confident): `Person`, `Organization`, `Skill`, `Location`, `Date`, `Amount`, `Clause`.

## Relationships

### `(Document)-[:MENTIONS]->(Entity)`
**Properties:**
- `chunk_id` (string)
- `evidence_span` (string)
- `confidence` (float)
- `edge_key` (string, deterministic)

### `(Entity)-[:ALIAS_OF]->(Entity)`
**Properties:**
- `confidence` (float)

### `(Person)-[:WORKED_AT]->(Organization)`
**Properties:**
- `start` (string)
- `end` (string)
- `role` (string)
- `confidence` (float)
- `chunk_id` (string)
- `edge_key` (string)

### `(Person)-[:HAS_SKILL]->(Skill)`
**Properties:**
- `confidence` (float)
- `chunk_id` (string)
- `edge_key` (string)

### `(Document)-[:HAS_FIELD]->(Entity)`
**Properties:**
- `key` (string)
- `value` (string)
- `confidence` (float)
- `chunk_id` (string)
- `edge_key` (string)

### `(Document)-[:RELATED_TO]->(Document)`
**Properties:**
- `edge_key` (string)

## Ingestion Notes
- Documents and entities are upserted.
- Relationship uniqueness is enforced via `edge_key` (deterministic key per relationship + chunk).
- `graph_version` is stored on `Document` to prevent duplicate ingests.

## Provenance Rules
- All relationships that feed the QA system must include a `chunk_id` so the fact can be grounded to evidence.
- Facts without `chunk_id` are excluded from answer generation.
