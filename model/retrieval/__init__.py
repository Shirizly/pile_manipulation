"""Retrieval-based transition model (EXP-0059).

Genesis-free (torch / numpy / scipy only), a new `model/` family per project
convention (ARCHITECTURE.md's "new model family" rule -- multi-file code, not
a parameter change over an existing baseline).

See `docs/experimental_design/retrieval_based_modeling.md` for the design
rationale. This package is the v0 groundwork: push-frame canonicalisation
(`frame.py`), a flat per-object transition bank over DS-0008/DS-0010
(`bank.py`), and particle-in/particle-out reference predictors
(`predictor.py`) -- persistence and a naive whole-bank 1-NN. It is
deliberately the simplest member of the design doc's representation x
retrieval x delta cell (R2 + K1 + D1): no interaction-support hierarchy, no
kernel/local-linear correction, no learned embedding.
"""
from .frame import (
    push_frame_basis, world_to_push_frame, push_frame_to_world,
    push_angle, yaw_from_quat, wrap_angle, tray_corners_push_frame,
)
from .bank import TransitionBank
from .distance import DistanceConfig, topk_search, query_points_and_weights, bank_points_and_weights
from .interaction import interaction_set, recall_precision
from .predictor import (PersistencePredictor, NearestTransitionPredictor,
                        RetrievalPredictor, AGGREGATIONS)

__all__ = [
    "push_frame_basis", "world_to_push_frame", "push_frame_to_world",
    "push_angle", "yaw_from_quat", "wrap_angle", "tray_corners_push_frame",
    "TransitionBank", "PersistencePredictor", "NearestTransitionPredictor",
    "DistanceConfig", "topk_search", "query_points_and_weights", "bank_points_and_weights",
    "RetrievalPredictor", "AGGREGATIONS", "interaction_set", "recall_precision",
]
