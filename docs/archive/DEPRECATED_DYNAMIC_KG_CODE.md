# Deprecated Dynamic KG Code (Archived)

Date: 2026-05-31

The following files were introduced for the dynamic/on-the-fly KG experiment and are now archived.
They are intentionally disabled by default to prevent accidental reuse.

## Archived Files
- `run_graph_dynamic.sh`
- `run_qagnn__csqa_dynamic.sh`
- `scripts/generate_dynamic_graph_from_llm.py`

## How They Are Marked
- Each file now contains a `DEPRECATED` banner at the top.
- Each file exits immediately unless:
  - `ALLOW_DEPRECATED_DYNAMIC_KG=1`

## Reason for Archival
- Severe regression versus baseline QA-GNN.
- Dynamic graphs frequently degenerated into near-empty/single-edge structures.
- High dependence on unstable network-proxy behavior during graph generation.

## Recommendation
- Use baseline static graph pipeline (`data/csqa/graph/*.pk`) for any further experiments.
