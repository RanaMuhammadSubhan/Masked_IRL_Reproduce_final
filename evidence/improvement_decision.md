# Evidence-based improvement decision

The mandatory robustness experiment found that Explicit Mask retained higher adapted accuracy than original Masked IRL at severe mask corruption. This result motivated the separately frozen confidence-aware extension. No primary setting or completed run was replaced.

The candidate uses demonstration/training-state empirical-CDF distance to soften declared irrelevance. Its formula and complete 15-run grid were fixed before extension outcomes; the clean lambda controls add six runs. All planned runs passed independent verification.

## Complete outcome comparison

| p | Endpoint | Candidate minus original Masked (pp) | Candidate minus Explicit (pp) |
|---|---|---:|---:|
| 0 | Few-shot adapted | -1.27 +/- 0.31 | -2.24 +/- 1.89 |
| 0 | Unadapted, exploratory | -1.44 +/- 0.37 | -0.63 +/- 3.98 |
| 0.05 | Few-shot adapted | +0.40 +/- 0.59 | +1.79 +/- 2.92 |
| 0.05 | Unadapted, exploratory | +1.96 +/- 0.32 | +6.43 +/- 2.09 |
| 0.1 | Few-shot adapted | +4.18 +/- 0.11 | +0.57 +/- 0.20 |
| 0.1 | Unadapted, exploratory | +2.82 +/- 1.53 | +6.53 +/- 1.04 |
| 0.2 | Few-shot adapted | +16.23 +/- 2.79 | +2.13 +/- 1.36 |
| 0.2 | Unadapted, exploratory | +9.39 +/- 2.12 | +7.43 +/- 1.14 |
| 0.3 | Few-shot adapted | +14.29 +/- 1.15 | +6.73 +/- 2.12 |
| 0.3 | Unadapted, exploratory | +12.99 +/- 0.91 | +9.91 +/- 2.25 |

Values are paired mean +/- SE across three fixed training seeds. All five probabilities and both endpoints are shown; there is no selection of a favorable subset.

The candidate has a positive adapted mean difference at 4/5 conditions against original Masked IRL and 4/5 against Explicit Mask. These counts are descriptive and are not independent hypothesis tests or a universal-superiority claim.
The candidate has a positive unadapted mean difference at 4/5 conditions against original Masked IRL and 4/5 against Explicit Mask. These counts are descriptive and are not independent hypothesis tests or a universal-superiority claim.

## What the confidence weights actually do

The following diagnostics use oracle truth only after fitting. They never feed the candidate or checkpoint selection. Entries summarize the 40 pretraining identities. Retained penalty mass is the mean fraction across three seeds. Mean confidence weights on false/correct irrelevance are pooled using their coordinate counts; lower means a weaker constraint.

| p | Retained penalty mass | Weight on false irrelevance | Weight on correct irrelevance |
|---|---:|---:|---:|
| 0 | 0.629 | n/a (no such bits) | 0.629 |
| 0.05 | 0.624 | 0.448 | 0.627 |
| 0.1 | 0.620 | 0.437 | 0.629 |
| 0.2 | 0.609 | 0.420 | 0.630 |
| 0.3 | 0.592 | 0.412 | 0.628 |

Confidence is not calibrated causal relevance. Marginal state-distribution shifts can arise from correlations; ten demonstrations are noisy. The method cannot repair false relevance declarations.

## Decision and claim boundary

Retain the candidate as a completed exploratory extension with all outcomes. Keep lambda=10 and the original mask objective as the primary paper-aligned reconstruction. The candidate is our proposed variant, not part of the original paper, and its observed scores do not retroactively redefine the baseline.

The candidate changes total regularization mass as well as coordinate weights. Clean lambda controls show sensitivity to strength but are not a matched-mass control under corruption. The present evidence cannot uniquely attribute a gain to identifying erroneous constraints.

The benchmark was already inspected when this method was designed. A fresh benchmark is needed for confirmatory improvement claims. Gaussian perturbations, trainable T5, corrected importance estimation, additional seeds and the candidate with saved LLM masks remain deferred.

The next focused study should compare the frozen candidate against a global penalty-mass-matched control at each corruption level, then test both on a fresh environment family with prespecified seeds. Do not change this completed formula using its test outcomes.
