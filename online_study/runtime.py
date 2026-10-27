"""Serving-only extensions without changing the sealed training implementation."""
from __future__ import annotations

import numpy as np

from reliability_cascade.history import history_candidate
from reliability_cascade.online_runtime import ExactLRU, post_fuse, pre_route


class ServingLRU(ExactLRU):
    def put_after_group(self, identity, value, observed=True):
        super().put_after_group(identity, value, observed)
        if observed and self.capacity:
            # A,B,A within one timestamp group must end in B,A, matching the
            # offline replay's last-encounter insertion order.
            self.pending.move_to_end((self.namespace, str(identity)))


def route_frozen(model, features, index, operation):
    if operation == "exact_bypass":
        if features.pre_names != model.pre_names:
            raise ValueError("Pre-route schema differs from frozen model")
        row = features.frame.iloc[index]
        if str(row.resource) != model.resource:
            raise ValueError("Wrong resource gate selected")
        applicable = bool(row.resource_applicable)
        eligible = applicable and bool(row.user_available) and bool(row.exact_available)
        return {"applicable": applicable, "eligible": eligible, "bypass": eligible,
                "probability": None,
                "history_prediction": float(history_candidate(features.frame, model.candidate_name)[index])}
    if operation == "no_bypass":
        # Remove the entire pre-classifier operation, retaining the frozen
        # post-gate and experts. Do not charge for a discarded classifier call.
        row = features.frame.iloc[index]
        return {"applicable": bool(row.resource_applicable), "eligible": False,
                "bypass": False, "probability": None,
                "history_prediction": float(history_candidate(features.frame, model.candidate_name)[index])}
    return pre_route(model, features, index)


def fuse_frozen(model, features, index, qwen, gates, user_known, operation):
    if operation == "no_post":
        return float(np.clip(qwen[index], 0., 1.))
    if operation == "no_trust" and model.config.trust_region_enabled:
        # Dataclasses are copied; the loaded checkpoint is never modified.
        from copy import copy
        from dataclasses import replace
        model = copy(model)
        model.config = replace(model.config, trust_region_enabled=False)
    return post_fuse(model, features, index, qwen, gates, user_known)
