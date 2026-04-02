"""
Shared constants used across all pipeline modules.

GROUP_COLS defines the 7-level hierarchy that identifies a unique cash-flow series.
All modules import from here — update in one place to change the hierarchy.
"""

GROUP_COLS = [
    "opcos",
    "entity_region",
    "entity_country",
    "entity_name",
    "account_name",
    "level2",
    "level5",
]

# Categorical columns that receive label encoding (same as group cols)
CAT_COLS = GROUP_COLS
