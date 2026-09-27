"""Build identical pairwise features for training and inference in batches.

Include text similarity, component agreement/conflict, and missingness.
Persist fitted feature state; never use entity IDs or labels as model features.
"""
