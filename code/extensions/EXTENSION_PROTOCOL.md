# Frozen extension plan after the mandatory robustness stage

Declared 7 October 2026 after all nine core and 24 additional corruption runs passed; before any extension result is produced. The six author-saved LLM runs retain priority and finish before extension training starts.

## Observation motivating the work

At p=.20, few-shot adapted scores were Explicit 64.03 +/- 0.69 and Masked 49.93 +/- 2.92 percent. At p=.30 they were 58.74 +/- 1.71 and 51.18 +/- 0.75. Both degrade; Masked IRL degrades more severely in this setting. This is retained rather than replaced by an expected narrative.

## Selected single-factor ablation: lambda

Run Masked IRL with lambda=1 and lambda=3, each on clean oracle masks with seeds 12345, 23456, 34567. Reuse the existing lambda=10 clean runs as the control. Only lambda changes. Six new runs; all schedules, data, initialization, demonstrations, masks, validation rules, encoder and importance estimator remain the same.

## Proposed method: uncertainty_aware_masked_irl

An incorrect irrelevance declaration creates an invariance constraint that can conflict with expert behavior. Use the existing ten demonstrations per training/adaptation identity to reduce confidence in that declaration when the observed state distribution differs from the training pool.

For each of the 19 state coordinates, take the mean over the 21 waypoints of each trajectory. Compute the two-sample empirical CDF distance D_j (the KS statistic) between the ten demonstration trajectory means and all 1,459 training trajectory means. No p-value or statistical-independence claim is made. The statistic is a bounded heuristic for distributional association, not calibrated causal relevance.

Given observed binary relevance b_j, define:

`soft_relevance_j = b_j + (1-b_j)*D_j`

The local regularizer therefore uses:

`(1-soft_relevance_j) = (1-b_j)*(1-D_j)`

Lambda stays 10. Already-declared relevant coordinates remain unpenalized. An irrelevant declaration is weakened in proportion to demonstration evidence. No true preference costs, oracle mask labels, nominal corruption probabilities, validation states or test states are used to compute D. Existing demonstrations are reused, with no new sampling.

Each identity's evidence uses only that identity's ten demonstrations and the training trajectory pool. Adaptation identities' evidence is used only during adaptation. Strict final identities receive no demonstrations or evidence fitting; their masks are recorded unchanged and are unused by Masked IRL inference. Projection, FiLM/reward, T5 freezing, optimizer, epoch/update counts, checkpoint selection and exclusive final evaluation remain unchanged.

This differs from the manuscript and is labeled OUR PROPOSED METHOD, not paper-faithful Masked IRL. It is designed after observing the original robustness results, so evaluations on the reused benchmark are exploratory, not an untouched confirmatory test. The original primary A/B endpoints remain frozen.

## Prespecified evaluation budget and ordering

After the saved-LLM stage, run the six lambda controls. Then run the proposed method at p=0,.05,.10,.20,.30, all three original seeds: 15 new runs. These use the same original binary corruption draws, data, demonstrations and initialization. Three new p=0 runs are necessary: the demonstration-based soft weighting also changes clean masks, so reusing the original clean learner would be invalid.

There are 21 extension runs (60 total completed learners if all succeed). This is a staged, selected matrix; Gaussian, trainable-T5, corrected-importance, extra seeds, and a proposed-method saved-LLM comparison are not included. Do not select a favorable subset of seeds, corruption levels or endpoints. Do not revise the formula based on extension test scores.

## Verification and limitations

Before training, test empirical-CDF distances analytically, bounds, constants, nonmutation, preservation of relevant bits, and soft-weight chunk values/gradients. Independently recompute confidence with a histogram-CDF implementation, without importing the training helper. Verify paired initialization and demonstrations against the frozen primary runs.

This heuristic can mistake correlated but irrelevant state variables for useful ones, miss relevance that leaves marginal trajectory means unchanged, and be noisy with ten demonstrations. It can only relax irrelevance penalties; it cannot correct false relevance labels. It also reduces total regularization strength, so any gain does not by itself establish that targeted confidence is better than a mass-matched global lambda. The clean lambda ablation is informative but is not that full matched-corruption control. A fresh benchmark and such a control remain necessary for stronger claims.
