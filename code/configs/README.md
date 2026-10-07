# Frozen configurations

Every executed config.json is retained byte-for-byte beside its run. evidence/run_registry.json maps all 60 runs to those configurations. No aggregate template replaces the actual executed settings. The protocol index here summarizes common choices.

Use the suite launchers to archive runnable source automatically. For direct single-run training, create work/single/source_archive/<SHA256>/<filename> for each .py file in code/reproduction before running; use scripts/archive_sources.py --source code/reproduction --output work/single. Extension direct runs also need the frozen EXTENSION_PROTOCOL.md at the output-root scripts/ location. The supplied extension suite handles this automatically.
