# Implementation decisions

M denotes src/models/reward_learning/masked_rl_lang_states_input.py. Full accepted source tracing is in first_checkpoint_source_audit.md.

## Freeze T5

**Decision:** Freeze T5

**Paper evidence:** IV-D describes a frozen pretrained encoder

**Released-code evidence:** masked_rl_lang_states_input.py:68–69 freeze commented; optimizer includes T5

**V3 choice:** Freeze pinned T5; train projection and reward

**Reason:** Recover manuscript representation boundary

**Consequence:** Different from V2 trainable encoder

## Lambda 10

**Decision:** Lambda 10

**Paper evidence:** Appendix B simulation lambda 10

**Released-code evidence:** maskedrl.yaml:20–21 lambda 1

**V3 choice:** Primary lambda 10; separate 1/3 ablation

**Reason:** Keep fidelity separate from empirical tuning

**Consequence:** Lambda 1 better here does not replace primary

## Uniform perturbations

**Decision:** Uniform perturbations

**Paper evidence:** IV-B local perturbation distribution

**Released-code evidence:** masked_rl_lang_states_input.py:508–520 Gaussian

**V3 choice:** One Uniform(0,1) draw per demo/time/coordinate

**Reason:** Recover stated perturbation

**Consequence:** Monte Carlo count is an explicit implementation choice

## Local loss aggregation

**Decision:** Local loss aggregation

**Paper evidence:** IV-B local mask equation

**Released-code evidence:** M:522–539 absolute joint trajectory change

**V3 choice:** Absolute before sum over T,D; mean B only

**Reason:** Avoid time cancellation and altered scale

**Consequence:** Lambda depends on aggregation convention

## Schedule and batch

**Decision:** Schedule and batch

**Paper evidence:** Figure 3; Appendix B

**Released-code evidence:** train.py:458–463 released mismatch

**V3 choice:** 1000+100; configured 512; actual 400/300

**Reason:** Match recoverable paper schedule

**Consequence:** 1100 updates, not 512 examples per epoch

## Adaptation learning rate

**Decision:** Adaptation learning rate

**Paper evidence:** Pretraining LR 0.001; separate adaptation constant not independently asserted

**Released-code evidence:** M:672 uses pretraining LR ×0.1

**V3 choice:** 0.001 then 0.0001; reset Adam

**Reason:** Preserve source-supported fine-tune relationship

**Consequence:** Adapted endpoint intentionally receives gradients

## Retain importance estimator

**Decision:** Retain importance estimator

**Paper evidence:** Estimator details insufficient to remove ambiguity

**Released-code evidence:** M:496–505 and 706–715 cross-sample broadcasting

**V3 choice:** Algebraically equivalent log-space cross-sample expression

**Reason:** Do not silently repair primary learner

**Consequence:** Partially paper-faithful; corrected estimator deferred

## Separate endpoints

**Decision:** Separate endpoints

**Paper evidence:** Paper-style adaptation receives demonstrations

**Released-code evidence:** V2 selected with later test preference identities

**V3 choice:** Training-identity validation plus distinct 30 strict identities

**Reason:** Prevent misleading unseen/zero-shot labels

**Consequence:** Strict and adapted scores use different identities

## Frozen language cache

**Decision:** Frozen language cache

**Paper evidence:** Frozen encoder permits caching

**Released-code evidence:** Released mean includes padded positions

**V3 choice:** Key exact instruction plus batch padded length

**Reason:** Preserve released language representation

**Consequence:** Kernel-order tolerance recorded; token lookup separately tested

## Replacement dataset

**Decision:** Replacement dataset

**Paper evidence:** Original central simulation assets unavailable

**Released-code evidence:** Learning code available but trajectories/resources missing

**V3 choice:** Reuse fixed controlled simulation_v2

**Reason:** Enable meaningful comparative experiment

**Consequence:** No exact numerical/environment reproduction

## Soft mask candidate

**Decision:** Soft mask candidate

**Paper evidence:** Not a paper method

**Released-code evidence:** Baseline stress tests exposed degradation

**V3 choice:** m=b+(1-b)D from train-only CDF association

**Reason:** Explore false-irrelevance relaxation

**Consequence:** Exploratory; mass confound; no false-relevance repair
