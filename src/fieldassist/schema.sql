CREATE EXTENSION IF NOT EXISTS vector;

-- One row per document revision. A manual and the bulletins that reissue its sections are separate
-- documents; which text is current is decided per section, in chunks.active.
CREATE TABLE IF NOT EXISTS documents (
    doc_id        text NOT NULL,
    revision      text NOT NULL,
    kind          text NOT NULL CHECK (kind IN ('manual', 'bulletin')),
    title         text NOT NULL,
    product_line  text NOT NULL,
    published     date NOT NULL,
    content_hash  text NOT NULL,
    ingested_at   timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (doc_id, revision)
);

-- A bulletin that reissues a manual section supersedes it, whatever order the two were ingested in.
CREATE TABLE IF NOT EXISTS supersessions (
    by_doc_id     text NOT NULL,
    doc_id        text NOT NULL,
    section_id    text NOT NULL,
    PRIMARY KEY (by_doc_id, doc_id, section_id)
);

CREATE TABLE IF NOT EXISTS chunks (
    chunk_id       text PRIMARY KEY,
    doc_id         text NOT NULL,
    revision       text NOT NULL,
    section_id     text NOT NULL,
    section_title  text NOT NULL,
    ordinal        int  NOT NULL,
    product_line   text NOT NULL,
    text           text NOT NULL,
    embedding      vector(768) NOT NULL,
    active         boolean NOT NULL DEFAULT true,
    superseded_by  text,
    tsv            tsvector GENERATED ALWAYS AS (to_tsvector('english', section_title || ' ' || text)) STORED,
    FOREIGN KEY (doc_id, revision) REFERENCES documents (doc_id, revision) ON DELETE CASCADE
);

-- Partial HNSW index: a filtered query on active rows uses it directly instead of post-filtering
-- an index scan over everything and coming back with fewer than k rows.
CREATE INDEX IF NOT EXISTS chunks_embedding_active
    ON chunks USING hnsw (embedding vector_cosine_ops) WHERE active;
CREATE INDEX IF NOT EXISTS chunks_embedding_all ON chunks USING hnsw (embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS chunks_tsv ON chunks USING gin (tsv);
CREATE INDEX IF NOT EXISTS chunks_doc ON chunks (doc_id, section_id);

CREATE TABLE IF NOT EXISTS eval_runs (
    run_id        text PRIMARY KEY,
    config_name   text NOT NULL,
    config        jsonb NOT NULL,
    model_image   text NOT NULL,
    golden_hash   text NOT NULL,
    metrics       jsonb NOT NULL,
    created_at    timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS promotions (
    id            bigserial PRIMARY KEY,
    environment   text NOT NULL,
    run_id        text NOT NULL REFERENCES eval_runs (run_id),
    baseline_run  text,
    decision      text NOT NULL CHECK (decision IN ('pass', 'block')),
    reasons       jsonb NOT NULL,
    model_image   text NOT NULL,
    created_at    timestamptz NOT NULL DEFAULT now()
);
