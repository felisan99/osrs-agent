-- Initial schema (SPEC §9). Plain Postgres + pgvector: runs unchanged on Homebrew Postgres,
-- the Supabase CLI's local database and a hosted Supabase project.
--
-- The vector dimension is fixed here (384: the default EMBEDDING_MODEL, see app/config.py).
-- `scripts.ingest load --force` re-creates `documents.embedding` and its index for another
-- EMBEDDING_DIM and re-ingests; `index_meta` (a later migration) records which model built it.

create extension if not exists vector;

create table documents (
  id bigserial primary key,
  page_title text not null,
  section text,
  page_type text not null
    check (page_type in ('quest', 'mmg', 'skill_guide', 'item', 'npc', 'other')),
  url text not null,
  content text not null,
  metadata jsonb not null default '{}',
  embedding vector(384) not null,
  tsv tsvector generated always as (to_tsvector('english', content)) stored
);

-- Re-ingesting a page deletes its chunks by title first, so this index keeps that cheap.
create index documents_page_title_idx on documents (page_title);
create index documents_embedding_idx on documents using hnsw (embedding vector_cosine_ops);
create index documents_tsv_idx on documents using gin (tsv);
create index documents_page_type_idx on documents (page_type);

-- Structured money-making guides (SPEC §8). `data` holds the validated MoneyMakingGuide model.
create table moneymaking_guides (
  id bigserial primary key,
  page_title text unique not null,
  data jsonb not null,
  ingested_at timestamptz not null default now()
);

-- Raw Jev probabilities, so thresholds can be tuned without calling Jev again (SPEC §9, §12).
-- Written ONLY in eval/debug mode, never with player data or the user's key.
create table jev_judgments (
  id bigserial primary key,
  run_id text not null,
  kind text not null check (kind in ('route', 'relevance', 'sufficiency')),
  input_ref text,
  probs jsonb not null,
  created_at timestamptz not null default now()
);
create index jev_judgments_run_idx on jev_judgments (run_id, kind);
