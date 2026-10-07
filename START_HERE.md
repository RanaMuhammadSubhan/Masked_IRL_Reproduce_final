# Masked IRL V3 professor submission

This is a **partially paper-faithful reconstruction on a controlled replacement simulation dataset**, with an exploratory robustness extension.

Read in order:
1. report/Masked_IRL_V3_Professor_Report.pdf — unchanged short report.
2. report/Masked_IRL_Reproduction_Implementation_and_Learnings.docx — implementation, reasoning, problems and lessons.
3. README.md — scope and exact commands.
4. results/tables/ — canonical per-seed, aggregate and paired results.
5. code/ and verification/ — runnable implementation and independent audit.

Core adapted means: LC-RL 65.07%, Explicit Mask 70.38%, Masked IRL 69.41%. Explicit has the highest adapted mean; Masked has the highest strict zero-shot mean (66.26%). There is no universal winner. Tables/report include three-seed SEs.

Original simulation assets were unavailable. This is not an exact numerical paper reproduction or a physical-robot result. The uncertainty-aware method improves high-corruption means but is exploratory, loses clean-condition accuracy and needs a fresh benchmark plus mass-matched controls.

From the extracted root, with Python and NumPy installed:
```sh
python verification/verify_submission.py --package .
```
The command checks all 60 learners without training. See environment/ENVIRONMENT_README.md for setup. The fixed dataset is included; pinned T5 weights must be downloaded separately only for training/checkpoint inference.
