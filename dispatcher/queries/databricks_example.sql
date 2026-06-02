-- Example Databricks SQL query for DatabricksWatcher.
--
-- The query MUST return exactly these three columns:
--   slide_id      -- unique identifier for the slide
--   slide_path    -- file path or cloud URI (e.g. s3://bucket/slides/1234.svs)
--   oncotree_code -- cancer type code; use '' if not applicable
--
-- Slides already known to the dispatcher (in its StateStore) are skipped automatically,
-- so it is safe to re-run this query on every poll cycle without deduplication logic.

SELECT
    s.slide_id       AS slide_id,
    s.slide_path     AS slide_path,
    s.oncotree_code  AS oncotree_code
FROM your_catalog.your_schema.slides s
WHERE s.slide_path IS NOT NULL
