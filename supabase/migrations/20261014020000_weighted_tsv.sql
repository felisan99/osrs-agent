-- Full-text search weights the chunk header above the body (SPEC §7 `retrieve`).
--
-- Every chunk starts with "Page title > Section path" (app/ingest/chunk.py). Giving those words
-- weight A (body: D) makes a question that names a page ("Monkey Madness II", "Grand Exchange")
-- rank that page's chunks first. Measured on the eval set with the English rewrite in place:
-- the answer chunk lands in the top 5 for 15/18 questions instead of 9/18, and every answerable
-- question finds it in the top 20 (16/18 before).
--
-- A generated column cannot be altered in place: drop it (its GIN index goes with it) and add it
-- back. It is recomputed from `content` by Postgres, so no re-ingestion is needed.

alter table documents drop column tsv;
alter table documents add column tsv tsvector generated always as (
  setweight(to_tsvector('english', split_part(content, E'\n', 1)), 'A')
  || setweight(to_tsvector('english', content), 'D')
) stored;
create index documents_tsv_idx on documents using gin (tsv);
