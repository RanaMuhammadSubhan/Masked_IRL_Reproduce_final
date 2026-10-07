# Limitations

- Partially paper-faithful reconstruction on a controlled replacement simulation dataset. Original main simulation assets are missing; no exact numerical, original environment or physical robot reproduction is claimed.
- Three seeds on one fixed dataset; SE represents seed variability, not cross-environment uncertainty.
- Released cross-sample importance estimator retained; its numerical stabilization preserves its behavior.
- Independent mask-bit flips simplify structured semantic errors. Saved masks have incomplete historical LLM provenance; no fresh API calls.
- The uncertainty-aware method is exploratory, designed after baseline inspection, and has no fresh confirmatory benchmark. Its marginal association score is heuristic and does not fix false relevance.
- Corruption and soft masks change penalty mass. Mass-matched controls are required for causal attribution.
- Independent verification audits saved predictions and execution evidence. It does not independently rerun every checkpoint or certify causal claims.
- Training/inference require a compatible CUDA/PyTorch environment and separately cached pinned T5 weights. The fixed replacement dataset is included.

Deferred: fresh benchmark; five seeds; mass-matched controls; candidate with saved LLM masks; corrected estimator; noise and encoder ablations; structured errors; physical robot studies.
