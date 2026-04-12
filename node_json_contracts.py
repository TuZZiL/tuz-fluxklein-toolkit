"""
Pure JSON contract helpers for node-backed hidden fields.
"""

from __future__ import annotations

import json
import logging

logger = logging.getLogger(__name__)


def warn_invalid_json(node_label, field_name, exc):
    logger.warning(f"[{node_label}] Invalid {field_name} JSON ignored: {exc}")


def parse_layer_strengths_json(raw_value, node_label):
    if not raw_value:
        return {}
    try:
        raw = json.loads(raw_value)
    except Exception as exc:
        warn_invalid_json(node_label, "layer_strengths", exc)
        return {}
    if not isinstance(raw, dict):
        logger.warning(f"[{node_label}] layer_strengths must be a JSON object; ignoring value")
        return {}
    if "db" not in raw and "sb" not in raw:
        logger.warning(f"[{node_label}] layer_strengths missing both 'db' and 'sb'; ignoring value")
        return {}
    return raw


def parse_slot_data_json(raw_value, node_label):
    if not raw_value:
        return []
    try:
        raw = json.loads(raw_value)
    except Exception as exc:
        warn_invalid_json(node_label, "slot_data", exc)
        return None
    if not isinstance(raw, list):
        logger.warning(f"[{node_label}] slot_data must be a JSON list; ignoring value")
        return None
    return raw

