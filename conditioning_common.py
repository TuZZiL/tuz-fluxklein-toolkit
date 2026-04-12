from __future__ import annotations

from typing import Any, Iterable, Optional

import torch


def clone_meta(meta: dict[str, Any]) -> dict[str, Any]:
    cloned = dict(meta)
    refs = meta.get("reference_latents")
    if isinstance(refs, (list, tuple)):
        cloned["reference_latents"] = [ref.clone() if isinstance(ref, torch.Tensor) else ref for ref in refs]
    elif isinstance(refs, torch.Tensor):
        cloned["reference_latents"] = refs.clone()
    return cloned


def _mask_to_tensor(mask: Any) -> Optional[torch.Tensor]:
    if mask is None:
        return None
    if isinstance(mask, torch.Tensor):
        return mask
    if isinstance(mask, (list, tuple)):
        try:
            return torch.as_tensor(mask)
        except Exception:
            return None
    return None


def find_attention_mask(tokenized: Any) -> Optional[torch.Tensor]:
    if isinstance(tokenized, dict):
        if "attention_mask" in tokenized:
            return _mask_to_tensor(tokenized["attention_mask"])
        for value in tokenized.values():
            found = find_attention_mask(value)
            if found is not None:
                return found
        return None
    if isinstance(tokenized, (list, tuple)):
        for value in tokenized:
            found = find_attention_mask(value)
            if found is not None:
                return found
    return None


def active_end_from_attention_mask(attn_mask: Any, seq_len: int) -> Optional[int]:
    mask = _mask_to_tensor(attn_mask)
    if mask is None:
        return None
    if mask.ndim == 0:
        return None
    if mask.ndim > 1:
        mask = mask[0]
    mask = mask.reshape(-1)
    if mask.numel() == 0:
        return None
    positives = (mask > 0).nonzero(as_tuple=False)
    if positives.numel() == 0:
        return None
    return min(int(positives[-1].item()) + 1, seq_len)


def detect_active_end(meta: dict[str, Any], seq_len: int, override: int = 0, fallback: int = 77) -> int:
    if override > 0:
        return max(0, min(int(override), seq_len))
    active_end = active_end_from_attention_mask(meta.get("attention_mask"), seq_len)
    if active_end is not None:
        return active_end
    return min(seq_len, fallback)


def detect_active_slice(
    meta: dict[str, Any],
    seq_len: int,
    *,
    skip_bos: bool = True,
    override: int = 0,
    fallback: int = 77,
) -> tuple[int, int]:
    end = detect_active_end(meta, seq_len, override=override, fallback=fallback)
    start = 1 if skip_bos and end > 1 else 0
    start = min(start, end)
    return start, end


def _extract_reference_latents_from_value(value: Any) -> list[torch.Tensor]:
    if value is None:
        return []
    if isinstance(value, torch.Tensor):
        return [value]
    if isinstance(value, (list, tuple)):
        return [ref for ref in value if isinstance(ref, torch.Tensor)]
    if hasattr(value, "cond"):
        return _extract_reference_latents_from_value(value.cond)
    return []


def get_reference_latents(meta: dict[str, Any]) -> list[torch.Tensor]:
    refs = meta.get("reference_latents")
    if refs is not None:
        return _extract_reference_latents_from_value(refs)

    model_conds = meta.get("model_conds", {})
    if isinstance(model_conds, dict):
        refs = model_conds.get("ref_latents", None)
        if refs is not None:
            return _extract_reference_latents_from_value(refs)
    return []


def set_reference_latents(meta: dict[str, Any], refs: Iterable[torch.Tensor]) -> dict[str, Any]:
    new_meta = clone_meta(meta)
    new_meta["reference_latents"] = list(refs)
    return new_meta


def apply_preserve_blend(enhanced: torch.Tensor, original: torch.Tensor, preserve_original: float) -> torch.Tensor:
    preserve = float(max(0.0, min(1.0, preserve_original)))
    return enhanced * (1.0 - preserve) + original * preserve


def dampen_toward_neutral(value: float, neutral: float, preserve_original: float) -> float:
    preserve = float(max(0.0, min(1.0, preserve_original)))
    return neutral + (value - neutral) * (1.0 - preserve)


def reference_indices(reference_count: int, reference_index: int) -> list[int]:
    if reference_count <= 0:
        return []
    if reference_index is None or reference_index < 0:
        return list(range(reference_count))
    if reference_index >= reference_count:
        return []
    return [int(reference_index)]


def reference_token_spans(extra_options: dict[str, Any], reference_index: int = -1) -> list[dict[str, int]]:
    ref_tokens = extra_options.get("reference_image_num_tokens", [])
    if not ref_tokens:
        return []

    total_ref = sum(ref_tokens)
    spans = []
    tok_start = 0
    for idx, token_count in enumerate(ref_tokens):
        seq_start = -total_ref + tok_start
        seq_end = seq_start + token_count
        spans.append(
            {
                "index": idx,
                "total_ref": total_ref,
                "tok_start": tok_start,
                "tok_end": tok_start + token_count,
                "seq_start": seq_start,
                "seq_end": seq_end,
                "seq_end_idx": None if seq_end == 0 else seq_end,
                "num_ref_tokens": token_count,
            }
        )
        tok_start += token_count

    if reference_index is None or reference_index < 0:
        return spans
    if reference_index >= len(spans):
        return []
    return [spans[int(reference_index)]]

