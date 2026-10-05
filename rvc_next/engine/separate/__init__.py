"""Source separation on ``pymss`` roformer models: presets, chains and the separator adapter.

``presets`` imports nothing heavy, so the core can list and resolve presets in-process; ``runner``
imports pymss and torch and runs only inside the separation worker.
"""
