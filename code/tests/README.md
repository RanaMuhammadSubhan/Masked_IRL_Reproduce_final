# Numerical checks

Run `python -B code/reproduction/test_objective.py` (8 tests) and `python -B code/extensions/test_confidence.py` (7 tests). These test small analytical tensors and confidence calculations; they do not train a learner. Tests cover chunked values/gradients, local perturbation semantics, corruption pairing, cross-sample estimator equivalence and soft-weight behavior. The executed test evidence is retained under results/primary and results/extensions.
