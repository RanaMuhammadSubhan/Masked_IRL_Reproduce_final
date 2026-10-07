# Fixed replacement dataset

The consolidated simulation_v2 dataset is included once at code/upstream/reconstruction/datasets/simulation_v2/. It is replacement simulation data, not the unavailable original paper trajectories. No data was regenerated for this package.

3000 candidates, 784 rejected, 2216 retained; 29 usable scenes; 1459/359/398 train/validation/test trajectories; 19/5/5 scenes; 323/81/87 start-goal groups. Endpoint mismatch is zero. There are 21 waypoints, 121 raw values, 19 model-state dimensions and five preference features. Scaling uses only training extrema. The actual eight dataset hashes used by every run remain in its frozen config.json and the global manifest.

Included arrays: trajectories.npy, states.npy, features.npy, features_unscaled.npy, scene_ids.npy and start_goal_group_ids.npy, plus splits.json and feature_scaling.json. Dataset manifest, validation and endpoint audits accompany them. Thirty redundant per-scene candidate arrays are omitted. No external trajectory download is necessary.

preference_manifest.json is the original accepted 40/30/30 split and hash-ranked strict preference selection. strict_zero_shot_preferences.json is its readable export. Each run's protocol.json stores its actual demonstrations and masks. Run `python scripts/verify_dataset.py` for a hash/split/scaling/endpoint check and the full verifier for independent demonstration/mask reconstruction.
