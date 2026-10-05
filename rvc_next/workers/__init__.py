"""Worker entry points, each run as ``python -m rvc_next.workers.<name> <request.json>``.

Workers import the engine and the protocol only, never the core or a framework.
"""
