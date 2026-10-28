-- Which embedding model built the vectors in `documents`.
--
-- Two models with the same dimension (e.g. two 768-dim ones) produce vectors that Postgres accepts
-- side by side and that mean nothing to each other, so a changed EMBEDDING_MODEL would silently
-- wreck retrieval. `scripts.ingest load` records the model here; the API refuses to start when
-- its own EMBEDDING_MODEL differs.
create table index_meta (
  key text primary key,
  value text not null,
  updated_at timestamptz not null default now()
);
