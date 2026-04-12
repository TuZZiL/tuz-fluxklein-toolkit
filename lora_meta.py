"""
Runtime LoRA analysis helpers used by node code and tests.
"""

import math
from pathlib import Path
from collections import OrderedDict, defaultdict

try:
    from .flux_constants import N_DOUBLE, N_SINGLE
except ImportError:
    from flux_constants import N_DOUBLE, N_SINGLE

try:
    from .lora_compat import parse_lora_key
except ImportError:
    from lora_compat import parse_lora_key

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

try:
    from .safetensors_reader import bytes_to_floats, parse_json_field, read_header, read_tensor_bytes
except ImportError:
    from safetensors_reader import bytes_to_floats, parse_json_field, read_header, read_tensor_bytes

_ANALYSIS_CACHE = OrderedDict()
_ANALYSIS_CACHE_MAXSIZE = 32


def _cache_get(cache_key):
    cached = _ANALYSIS_CACHE.get(cache_key)
    if cached is not None:
        _ANALYSIS_CACHE.move_to_end(cache_key)
    return cached


def _cache_set(cache_key, value):
    _ANALYSIS_CACHE[cache_key] = value
    _ANALYSIS_CACHE.move_to_end(cache_key)
    while len(_ANALYSIS_CACHE) > _ANALYSIS_CACHE_MAXSIZE:
        _ANALYSIS_CACHE.popitem(last=False)
# ─────────────────────────────────────────────────────────────────────────────
# ARCHITECTURE DETECTION
# ─────────────────────────────────────────────────────────────────────────────

def detect_architecture(keys):
    k = "\n".join(keys).lower()
    if "double_blocks" in k or "single_blocks" in k:
        if "klein" in k or "flux2" in k or "9b" in k:
            return "FLUX.2 Klein"
        return "FLUX.1 / FLUX.2"
    if "transformer_blocks" in k and "unet" not in k:
        return "FLUX-like DiT"
    if "up_blocks" in k and "text_model" in k:
        return "SDXL"
    if "up_blocks" in k:
        return "SD 1.x/2.x"
    if "mmdit" in k:
        return "SD3"
    return "Unknown"

def detect_lora_type(keys):
    k = "\n".join(keys).lower()
    if "dora_scale" in k or "magnitude" in k:
        return "DoRA (weight-decomposed)"
    if "lokr" in k or "kron" in k:
        return "LoKr"
    if "loha" in k or "hada" in k:
        return "LoHa"
    if any(token in k for token in ("lora_down", "lora.down", "lora_a")):
        return "Standard LoRA"
    return "Unknown"

# ─────────────────────────────────────────────────────────────────────────────
# LAYER PARSING
# ─────────────────────────────────────────────────────────────────────────────

def layer_type(base_key: str):
    k = base_key.lower()
    if any(x in k for x in ["to_q", "attn_q", "q_proj"]):   return "attn_q"
    if any(x in k for x in ["to_k", "attn_k", "k_proj"]):   return "attn_k"
    if any(x in k for x in ["to_v", "attn_v", "v_proj"]):   return "attn_v"
    if any(x in k for x in ["to_out", "proj_out", "out_proj"]): return "attn_out"
    if any(x in k for x in ["ff", "mlp", "feed_forward"]):  return "ff/mlp"
    if "proj" in k:                                           return "proj"
    if "norm" in k:                                           return "norm"
    if "embed" in k:                                          return "embed"
    return "other"

# ─────────────────────────────────────────────────────────────────────────────
# WEIGHT STATS
# ─────────────────────────────────────────────────────────────────────────────

def compute_stats(arr):
    if arr is None or len(arr) == 0:
        return {}
    arr = arr.astype(np.float32) if HAS_NUMPY else arr
    finite = arr[np.isfinite(arr)] if HAS_NUMPY else arr
    if len(finite) == 0:
        return {"all_nan_or_inf": True}
    return {
        "mean":   float(np.mean(finite)),
        "std":    float(np.std(finite)),
        "min":    float(np.min(finite)),
        "max":    float(np.max(finite)),
        "l2":     float(np.sqrt(np.sum(finite ** 2))),
        "l1":     float(np.mean(np.abs(finite))),
        "nz_pct": float(100.0 * np.count_nonzero(finite) / len(finite)),
    }

def effective_rank(arr, shape):
    """Estimate effective rank via SVD on the reshaped down weight matrix."""
    if not HAS_NUMPY or arr is None:
        return None
    try:
        mat = arr.reshape(shape[0], -1).astype(np.float32)
        s = np.linalg.svd(mat, compute_uv=False)
        s = s[s > 1e-6]
        if len(s) == 0:
            return 0
        s_norm = s / s.sum()
        entropy = -float(np.sum(s_norm * np.log(s_norm + 1e-12)))
        return round(math.exp(entropy), 2)
    except Exception:
        return None

# ─────────────────────────────────────────────────────────────────────────────
# NODE API  — called by flux_lora_loader.py
# Returns structured per-layer ΔW data ready for strength computation.
# ─────────────────────────────────────────────────────────────────────────────

def analyse_for_node(path):
    """
    Run the full forensic analysis on a safetensors file and return a dict:
    {
      "db": {
          0: {"img": float, "txt": float},   # mean scaled ΔW per stream
          ...
      },
      "sb": {
          0: float,   # mean scaled ΔW
          ...
      },
      "rank":  int,
      "alpha": float | None,
      "layer_stats": [(base, dw_scaled_norm), ...]   # sorted by base key
    }
    """
    from collections import defaultdict

    path = Path(path).resolve()
    cache_key = (str(path), path.stat().st_size, path.stat().st_mtime_ns)
    cached = _ANALYSIS_CACHE.get(cache_key)
    if cached is not None:
        return cached
    meta, header, data_offset = read_header(path)

    # Build layers dict: base_key → {role: tensor_info}
    layers      = defaultdict(dict)
    layer_ranks = {}
    alpha_values = {}

    for key, info in header.items():
        if not isinstance(info, dict):
            continue
        base, role = parse_lora_key(key)
        layers[base][role] = info

    # Read alpha tensor values
    for base, roles in layers.items():
        if "alpha" in roles:
            try:
                raw = read_tensor_bytes(path, roles["alpha"], data_offset)
                arr = bytes_to_floats(raw, roles["alpha"].get("dtype", "F32"))
                if arr is not None and len(arr) > 0:
                    alpha_values[base] = float(arr[0])
            except Exception:
                pass
        dn = roles.get("lora_down") or roles.get("lora_A")
        if dn and "shape" in dn:
            layer_ranks[base] = dn["shape"][0]

    # Compute ΔW scaled norms per layer
    db_img = defaultdict(list)
    db_txt = defaultdict(list)
    sb      = defaultdict(list)
    all_layer_stats = []

    for base in sorted(layers.keys()):
        roles   = layers[base]
        dn_info = roles.get("lora_down") or roles.get("lora_A")
        up_info = roles.get("lora_up")   or roles.get("lora_B")

        if not dn_info or not up_info:
            continue
        if not HAS_NUMPY:
            continue

        try:
            dn_raw = read_tensor_bytes(path, dn_info, data_offset)
            up_raw = read_tensor_bytes(path, up_info, data_offset)
            dn_arr = bytes_to_floats(dn_raw, dn_info.get("dtype", "F32"))
            up_arr = bytes_to_floats(up_raw, up_info.get("dtype", "F32"))
        except Exception:
            continue

        try:
            dn_shape = dn_info["shape"]
            up_shape = up_info["shape"]
            dn_mat   = dn_arr.reshape(dn_shape[0], -1).astype(np.float32)
            up_mat   = up_arr.reshape(up_shape[0], -1).astype(np.float32)
            delta_w  = up_mat @ dn_mat
            dw_norm  = float(np.linalg.norm(delta_w, "fro"))

            alpha    = alpha_values.get(base)
            rank_val = layer_ranks.get(base, 1)
            scale    = (alpha / rank_val) if alpha else 1.0
            dw_scaled = dw_norm * scale
        except Exception:
            continue

        all_layer_stats.append((base, dw_scaled))

        # Classify into db_img / db_txt / sb
        clean = base
        for pfx in ("diffusion_model.", "transformer.", "unet."):
            if clean.startswith(pfx):
                clean = clean[len(pfx):]
                break
        # Remap diffusers/Musubi prefixes to native
        if clean.startswith("transformer_blocks."):
            clean = "double_blocks." + clean[len("transformer_blocks."):]
        elif clean.startswith("single_transformer_blocks."):
            clean = "single_blocks." + clean[len("single_transformer_blocks."):]
        parts = clean.split(".")
        try:
            if parts[0] == "double_blocks":
                idx  = int(parts[1])
                rest = ".".join(parts[2:])
                is_txt = any(x in rest for x in ("txt_", "add_q", "add_k", "add_v", "add_out", "ff_context"))
                (db_txt if is_txt else db_img)[idx].append(dw_scaled)
            elif parts[0] == "single_blocks":
                sb[int(parts[1])].append(dw_scaled)
        except (IndexError, ValueError):
            continue

    # Aggregate
    all_alphas = list(alpha_values.values())

    result = {
        "db": {
            i: {
                "img": float(np.mean(db_img[i])) if db_img.get(i) else None,
                "txt": float(np.mean(db_txt[i])) if db_txt.get(i) else None,
            }
            for i in range(N_DOUBLE)
        },
        "sb": {
            i: float(np.mean(sb[i])) if sb.get(i) else None
            for i in range(N_SINGLE)
        },
        "rank":        sorted(set(layer_ranks.values()))[0] if layer_ranks else 0,
        "alpha":       float(np.mean(all_alphas)) if all_alphas else None,
        "layer_stats": all_layer_stats,
    }
    _cache_set(cache_key, result)
    return result


if __name__ == "__main__":  # pragma: no cover
    try:
        from .lora_meta_cli import main
    except ImportError:
        from lora_meta_cli import main
    main()
