"""Namespace for the pipeline stages. Each stage also runs as a standalone
script via its own run.py; this package init lets tests and the experiment
harnesses import stage modules (e.g. ``from stages.stage3_metric import align``).
"""
