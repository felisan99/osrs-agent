-- Structured quest requirements (SPEC §7 quest_workflow compares them to the player's state
-- deterministically, so they are stored as data, not left as prose in `documents`).
-- `data` holds the validated QuestRecord: skills (+ boostable), quest points, required quests with
-- prerequisite depth, items to bring, recommendations.
create table quests (
  id bigserial primary key,
  page_title text unique not null,
  data jsonb not null,
  ingested_at timestamptz not null default now()
);
