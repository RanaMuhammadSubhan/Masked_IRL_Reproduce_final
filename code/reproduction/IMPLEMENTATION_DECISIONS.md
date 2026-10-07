# V3 implementation decisions - 7 October 2026

The 6 October design checkpoint is accepted and is not repeated. Existing V2 data, source and frozen submission remain unchanged. New code and runs are under execution_20261007.

## Endpoints and selection

- Primary A: paper_faithful_adapted_v3, the original 30 held-out identities, ten demonstrations each, 100 adaptation epochs after validation-selected pretraining. Report both pre-adaptation and few-shot adapted scores.
- Secondary B: strict_zero_shot_v3 uses the previously hash-ranked new 30 identities and the selected pretraining weights. Those identities supply no gradients, demonstrations or selection scores.
- Historical C: released_reconstructed_v2 stays unchanged.
- Pretraining selection uses only training identities in validation scenes every ten epochs, maximum mean pairwise accuracy, earliest exact ties. Adaptation uses fixed epoch 100, without further selection.
- All fitting and selection finish before an exclusive final-test bundle evaluates saved pre/post checkpoints. Before-adaptation performance refers to pre-adaptation weights, not permission to inspect test scores and tune adaptation.

## Learning behavior

Pretraining is 1,000 complete passes over 400 demonstrations, one Adam update per pass at configured batch 512. Adaptation is 100 passes over 300 demonstrations, one update per pass. Target batch 512 passed a full-512-example profile; actual final demo batches are 400 and 300, not padded/repeated in training. Pretraining contrast batch is 512; adaptation contrast batch is 300, following the released fine-tune loop.

Pretraining LR is 0.001. Adaptation LR is 0.0001: the verified released fine-tune relationship is pretraining LR times 0.1 (masked_rl_lang_states_input.py:672; meirl_lang_states_input.py:613). This relationship is repository-supported, not asserted as an independently specified manuscript value. Adam is reset for adaptation, as in the released method. Projection, FiLM and reward MLP update; T5 stays frozen.

The original FiLMRewardModel class is imported without source edits: 19-state/128-condition, hidden widths 128/256/128, initial ReLU then Tanh, float64 reward weights. The added language projection is float32, as released. T5-base is pinned to a9723ea7f1b39c1eae772870f3b547bf6ef7e6c1; tokenizer also pinned, local cache only, no API calls.

The pretrained backbone is eval-mode, requires_grad=False, excluded from both optimizers, checked for absent gradients at every step, and hashed over all state tensors before and after. Projection and reward layers remain trainable. Frozen representations are cached only by exact instruction AND the dynamically padded length of the original call. Mean pooling still includes padded positions. A full-batch direct/cache comparison is asserted per run. Tiny floating-point kernel-order differences are recorded, not claimed bitwise identical.

After the one-seed gate, CPU profiling found repeated tokenization solely to recover the padded length. A lookup optimization memoizes each instruction's exact token count, then takes the batch maximum. All 203 tested batches covering 100 instructions produced bitwise identical token IDs/attention masks; the actual pooling method also matched a token-sensitive mock encoder bitwise. The unchanged per-run real-T5 direct/cache assertion still executes. The first three seed-12345 runs and the already-started Masked runs for seeds 23456/34567 retain the original length lookup. Later jobs use the verified memoization. This changes neither encoder inputs, cache keys/missing-key ordering, pooling, objectives, RNG streams nor update counts. Both source versions and test evidence are archived. Timing for 100 repeated 400-text batches fell from about 5.98 seconds to 0.004 seconds for this CPU lookup only; this is not claimed as the full training speedup.

## Mask objective and numerical stability

The local mask estimator perturbs one irrelevant coordinate at a time with one Uniform(0,1) draw, takes absolute state-cost differences, sums over time/coordinates, and averages over demonstrations. Chunk losses backpropagate into the same update; there is no per-chunk optimizer step or extra averaging. Lambda 10 is exclusive to Masked IRL. Numerical tests verify chunked/unchunked values and gradients, scale, coordinate isolation, bounds, nonmutation and absence of time cancellation.

The importance estimator remains released_cross_sample. Its ratio mean is evaluated in log space using mean_ij exp(-c_i)/p_j = mean_i exp(-c_i) * mean_j(1/p_j), with detached weights. A numerical test checks equality of both value and gradient to the released expression. This stabilizes exponentials without replacing the estimator with elementwise ratios. The overall reconstruction is therefore partially paper-faithful, not an exact manuscript implementation.

## Paired randomness and corruption

Initialization, demo sampling, training order and perturbation RNG streams are explicit. The same seed gives the same initial projection/reward weights and demonstrations across methods/conditions. Corruption uses SHA256(seed,preference) to seed one vector of independent uniforms; thresholding at each p gives nested errors. Exactly the same corrupted masks are used across paired Explicit/Masked runs. Masks affect training and adaptation; Explicit Mask also uses them at inference. Source masks are never mutated. Core p=0 runs are reused rather than retrained.

Downstream corruption is a limitation test, not automatically a new algorithm. Any confidence-aware treatment must follow and be separately labeled. No lower-priority ablation is permitted to precede core and robustness completion.

## Execution gates and failures retained

Eight numerical tests passed. One initial profile stopped at a singleton shape mismatch because the author oracle function returns (1,19); the adapter now explicitly flattens a single mask to (19). That failed profile remains recorded. The successful full-512 masked update used about 1.11 GB allocated and 1.50 GB reserved GPU memory, with finite gradients and matching T5 checksums. Wall time of a first update includes initial language-cache work and is not an epoch-runtime prediction.

Source scripts/config hashes, dataset hashes, complete stdout/stderr, final predicted/ground-truth costs, per-epoch losses, validation arrays, pre/adapt checkpoints and registry events are saved for every run. Final-test failures are recorded and are not silently retried. No paid services are required.

One initial Explicit launch used an incorrect preference-manifest path and failed before model initialization/training. The actual path was located, its accepted SHA256 was checked, and a new run completed. The failed launch remains in the ledger. Later scheduler invocations check manifest existence and hash before launch. Corruption runs are labeled mask_corruption_robustness, never an improved algorithm; the early trainer metadata wording was corrected before corruption execution, without changing the learner.
