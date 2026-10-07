# V3 first execution checkpoint

Prepared 6 October 2026. Scope of this checkpoint: verification, source tracing, protocol design and draft configuration only. No V3 learner has been implemented and no V3 training has been launched.

## Required current-state report

**CURRENT GIT COMMIT:** the target source snapshot is `b52bc2e3f1ec5597360a74c2641283f6c984a76b`. This is an archive extraction, not a Git checkout; no current branch/HEAD can be asserted. Git is unavailable in the checked execution environment.

**CURRENT WORKING TREE:** existing original source, additive adapters and submission files were preserved. No reset, checkout, rename or baseline modification was performed. Git working-tree cleanliness is not applicable; identity was checked through recorded hashes instead.

**V2 RESULTS LOCATED:** original `Masked-IRL/reproduction/results/raw/reconstructed_*` run IDs in `v2_independent_audit/main_results.csv`; original report `Masked-IRL/reconstruction/report_v2/`.

**V2 DATASET LOCATED:** `Masked-IRL/reconstruction/datasets/simulation_v2/` under the existing Desktop project.

**UPSTREAM SOURCE HASH STATUS:** PASS, all 155 original archive files. Additionally, all 465 files in the freeze manifests match BOTH the frozen copies and current originals. Selected checkpoint metadata and saved predictions also passed independent verification. Frozen submission stays in `Final_Submission_20261006/`; new work stays in `Paper_Faithful_V3_20261006/`.

**PAPER-FAITHFUL CHANGES REQUIRED:** freeze the pretrained T5 backbone; set simulation lambda 10; use Uniform(0,1); implement the local per-state/per-coordinate masking equation; use paper LR 0.001 and batch 512; plan 1,000 pretraining epochs; resolve the meaning of the additional 100 adaptation epochs before enabling them.

**VALIDATION/TEST PROTOCOL ISSUE:** V2 selected on the same preference identities later used in test scenes. V3 will select on training identities in validation scenes. Separately, manuscript-style fine-tuning adapts to test preferences, which conflicts with the new requirement that final-test preferences never receive gradients. This requires distinct zero-shot and adaptation endpoints; it cannot be solved by renaming adapted preferences as unseen.

**ESTIMATED RUN MATRIX:** 9 oracle/LC-RL runs at three seeds; 6 additional author-saved LLM-mask runs; 24 additional corruption runs (p=0 reuses matching clean oracle runs); 6 lambda runs; 6 noise runs; 9 trainable-T5 ablation runs = 60 unique proposed training runs, before optional soft masks or extra seeds. No estimate of completed runs is implied.

## V2 independently verified again

| Method | Selected epoch | Seen test | Unseen test |
|---|---:|---:|---:|
| LC-RL | 60 | 68.63% | 68.14% |
| Oracle Masked IRL | 60 | 71.37% | 70.35% |
| Explicit Mask | 80 | 71.98% | 72.83% |

Train/validation/test counts remain 1,459/359/398 trajectories, 19/5/5 scenes and 323/81/87 groups. There are 657 validation and 748 test candidate pairs before preference-dependent tie exclusions. Endpoint maximum difference is zero. No cross-split trajectory, scene or group overlap; feature scaling matches train-only extrema. This audit did not regenerate trajectories or rerun collision simulation.

## Source locations

Paths below are relative to the unmodified `Masked-IRL/` snapshot. `M` abbreviates `src/models/reward_learning/masked_rl_lang_states_input.py`.

| Behavior | Verified source location |
|---|---|
| T5 backbone and added projection | M:60-82; commented freeze at 68-69; linear projection at 70 |
| Optimizer includes T5 | M:224; fine-tuning optimizer M:674-676 |
| LC-RL / Explicit backbone and optimizer | `src/models/reward_learning/meirl_lang_states_input.py`:68,211; fine-tuning 599-617; Explicit Mask uses this class with masked inputs |
| Lambda / noise scale | M:219-220; simulation `config/reward_learning/obj20_sg10_persg5/maskedrl.yaml`:20-21 |
| Gaussian noise and mask replication | M:508-520; fine-tuning M:719-727 |
| Trajectory-cost masking objective | M:522-539; fine-tuning M:728-735 |
| State costs summed across trajectory | M:297-318 |
| Importance-weight broadcasting | M:496-505; fine-tuning M:706-715 |
| Pretraining data assembly | `src/scripts/train.py`:278-305; invocation at 458 |
| Fine-tuning preference/data assembly | `src/scripts/train.py`:312-333; invocation and schedule at 462-463 |
| Fine-tuning implementation | M:658-738; LR factor 0.1 at 672; updated model and language parameters at 674-676 |
| Original preference split | `src/scripts/train.py`:139, 231-233; hardcoded sparse split path |
| V2 40/30 preference construction | `reproduction/scripts/train_reconstructed.py`:34-36 |
| V2 demonstration sampling | same adapter:37-46 |
| V2 checkpoint selection | same adapter:111-121 |
| V2 pairwise evaluation and saved costs | same adapter:47-53, 82-104; final call at 122 |
| Native pairwise evaluators | M:320-356; `src/models/reward_learning/meirl_lang_states_input.py`:307-344 |
| Clear-language author-saved mask lookup | `src/utils/feature_utils.py`:5-10, 307-314 |

Paper evidence: [manuscript v1](https://arxiv.org/html/2511.14565v1), Section IV-B (local mask equation), IV-D (encoder freezing), Figure 3 (phase schedule and preference counts), Appendix B (optimizer settings and lambda).

## Tensor-level masking difference

Released inputs have shape (B,T,D), with T=21 and D=19. Ten repeats produce (10B,T,D). Noise changes all irrelevant coordinates simultaneously. The network sums state costs over T, then averages the absolute differences of trajectory costs. Opposite signed changes at different times can cancel before the absolute value.

The manuscript objective instead sums absolute state-cost changes over time and individual irrelevant coordinates, averaging over demonstrations. The proposed estimator uses one Uniform(0,1) draw per (demo,time,coordinate), perturbing one coordinate at a time, summing over T and D and averaging over B. Absolute cost and reward differences agree under a sign reversal. One draw is an explicit Monte Carlo implementation choice; the manuscript does not specify its sampling count. Memory chunks must preserve that sum/mean scaling and gradients. Do not average over T or D without documenting a changed effective lambda.

The local mask equation is sufficiently explicit to implement. The separate importance-sampling estimator remains ambiguous: preserve the released cross-sample calculation in the first V3 comparison, record it, and reserve any correction for a named ablation. Consequently, the entire pipeline should be labeled **PARTIALLY PAPER-FAITHFUL** even if the mask equation matches.

## Pretraining and fine-tuning semantics

Pretraining aggregates demonstrations of training preference identities over training trajectories. Fine-tuning aggregates demonstrations of the held-out preference identities, also drawn from the training trajectory pool. It updates a shared reward/language model, rather than creating 30 independent per-preference models. The released fine-tuner samples contrast trajectories from the existing training pool, reduces LR by 0.1, and includes language parameters. `train.py` requests 1,000 simulation fine-tuning epochs, although the paper caption specifies 100.

Therefore, fine-tuned preferences are adapted/few-shot preferences, not never-trained zero-shot preferences. The fixed reconstructed pool can support a reconstructed adaptation experiment; original assets are not needed for that reconstruction. What is unresolved is the requested scientific endpoint, not technical access to demonstrations.

## Change matrix

| Setting | V2 | Paper | Planned V3 |
|---|---|---|---|
| Dataset | Reconstructed v2 | Original author simulation | Same fixed v2 arrays and scene split |
| T5 | Trainable | Frozen pretrained encoder | Freeze backbone; projection/FiLM/reward trainable |
| Mask lambda | 1 | 10 simulation | 10 for Masked IRL only |
| Perturbation | Gaussian, scale 1 | Uniform(0,1) | Configurable; Uniform(0,1) primary |
| Mask objective | Simultaneous trajectory MAE | Local coordinate/state absolute change | Local sum, explicit Monte Carlo estimator |
| Pretraining | 100 epochs | 1,000 epochs | 1,000 planned |
| Fine-tuning | None | 100 on 30 test preferences | Blocked pending zero-shot/adaptation endpoint decision |
| Learning rate | 0.0001 | 0.001 | 0.001 primary, sourced from paper |
| Batch size | 64 | 512 | 512 target; profile memory before execution |
| Train preferences | 40 | 40 | Same 40 |
| Validation identities | Same 30 as final test | Not recovered | Training 40 only on validation scenes |
| Final zero-shot identities | 30 used in V2 selection | 30 held out before adaptation | 30 newly designated identities, excluding V2 train and validation/test identities |
| Demonstrations | 10 / train preference | 10 | 10; fixed paired draw within each seed |
| Seeds | 12345 | Five, exact IDs unestablished | Predeclare 12345,23456,34567; optional extension 45678,56789 |
| LLM masks | Oracle only | Oracle and LLM variants | Saved author masks where key mapping validates |
| Evaluation | Corrected dropout/masking | Pairwise ordering | Preserve V2 tie and pair rules; new test lock |

## Corrected validation / final-test design

1. Keep the existing v2 trajectory arrays, scene IDs, split and train-only scaling fixed.
2. Train on the same 40 identities. Evaluate validation scenes using those 40 identities only; select mean pairwise accuracy every ten epochs, earliest tie wins. This explicitly changes the selection target from V2.
3. Predeclare 30 new final zero-shot preferences from ternary nonzero vectors excluding every V2 training and validation/test identity. Select deterministically by SHA256 ranking with a declared salt, independent of performance. Save exact vectors and author-mask mappings.
4. Finalize config and selected checkpoint before evaluating test scenes on the new final preference identities. Store frozen-config/checkpoint hashes, UTC timestamps and an exclusive test-start marker. Record FAILED if evaluation fails; never silently retest or overwrite.
5. Report seen-preference test performance alongside zero-shot performance only at that locked final evaluation.
6. Prior V2 results exposed the same test scenes. This correction avoids reusing preference identities in model selection but cannot make the entire project prospectively blind to its historical benchmark. Disclose that limitation; do not claim a wholly new unseen dataset.
7. The proposed separate paper-style adaptation endpoint would adapt using old V2 held-out preferences and training scenes, then evaluate those adapted identities on test scenes with a distinct label. It must never be merged with primary zero-shot scores. This remains a proposed resolution awaiting the user's endpoint preference.

## Saved LLM masks

The author clear-mask file `config/data_split_config/theta_to_pred_mask_sdim19.json` contains 242 preference-keyed masks. All existing 40 training and 30 V2 held-out identities map to binary 19-dimensional masks without missing keys. The source clear-language lookup uses those exact keys. New final identities will receive the same structural/key checks in the configuration audit. This supports `author_saved_llm_mask`, without asserting the unavailable historical model/version/round provenance or regenerating masks with a modern API. Ambiguous demonstration-specific artifacts are not interchangeable with this clear-instruction mapping.

## V3 implementation plan

**Files to modify:** none of the original source, V2 adapters/runs, dataset, submission ZIP or frozen reports. New V3 files may be revised within their separate directory before config freeze.

**Files to create:** `reproduction/configs/paper_faithful_v3/protocol.draft.json`, exact preference/mask manifest, decision log and source map at this checkpoint. Later: a separate trainer/objective module, meaningful numerical tests, V3 independent verifier, one scheduler for seeds/corruption/ablations, append-only registry and aggregation/report scripts. Avoid multiple redundant wrappers.

**Paper-faithful changes:** backbone freeze with all-parameter checksums, optimizer exclusion and no-gradient checks; lambda 10; uniform perturbations; local mask equation; paper LR/batch/pretraining budget. Pin the existing T5 revision. Retain and disclose unresolved estimator behavior.

**Protocol corrections:** train-identity validation, new final preference identities, explicit one-shot test lock, matched mask corruption and saved demonstration/mask identities. No generated trajectory changes across seeds.

**Expected experiment count:** 9 primary oracle/LC-RL runs; 6 saved-LLM variants; 24 additional oracle-mask corruption runs = 39 core runs at three seeds. Add 6 lambda, 6 perturbation and 9 T5 ablations = 60. A five-seed version is 65 core / 100 including these ablations. Optional soft-mask experiments are additional and not predeclared as completed. Corruption is a robustness experiment, not by itself an implemented algorithmic improvement; a claimed new-method improvement would require the optional soft-mask variant or another separately justified change.

**Potential blockers:** zero-shot/fine-tuning conflict; estimator ambiguity; exact historical LLM provenance unavailable; frozen T5 mean-pooling behavior depends on padding, so embeddings must not be cached under an inequivalent pooling scheme; batch-512 memory and local-mask computation require profiling. Do not silently reduce batch size or rescale the objective.

**Estimated GPU workload category:** substantial staged research workload (tens of training jobs), not submission-only formatting. Each run plans 1,000 pretraining epochs, with any adaptation separately resolved. With 400 demonstrations and batch 512, one full demo pass can be one update; it is not simply ten times the 700-step V2 schedule. Local coordinate perturbations increase reward-network work. Estimate wall time only after the first valid profiled run; use one heavy run per GPU, initially sequential method gates, then two-GPU scheduling.

## Gate and current status

**STATUS: PARTIALLY PAPER-FAITHFUL - DESIGN CHECKPOINT, NOT IMPLEMENTED.** The draft configuration is explicitly non-runnable. V2 verification passed. No T5-freezing runtime claim, V3 score, multi-seed result, or downstream improvement result is claimed. Implementation stops here to report the requested first checkpoint and resolve the scientific endpoint conflict under the prompt's stop condition.
