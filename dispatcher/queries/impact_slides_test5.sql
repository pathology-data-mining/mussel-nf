-- Test query: 5 slides only. Do not use in production.
SELECT
    m.image_id      AS slide_id,
    i.path          AS slide_path,
    m.ONCOTREE_CODE AS oncotree_code
FROM cdsi_eng_phi.pdm_base_tables.impact_matched_slides m
JOIN cdsi_eng_phi.pdm_base_tables.slide_inventory i
  ON m.image_id = i.image_id
WHERE i.path IS NOT NULL
  AND i.size >= 10000000
  AND i.source IN ('ECS2')
  AND m.ONCOTREE_CODE IS NOT NULL
  AND m.ONCOTREE_CODE != ''
LIMIT 5
