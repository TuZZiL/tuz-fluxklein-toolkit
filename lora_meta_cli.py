#!/usr/bin/env python3
"""
CLI reporting wrapper for LoRA forensic analysis.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

try:
    from .lora_meta import (
        HAS_NUMPY,
        bytes_to_floats,
        compute_stats,
        detect_architecture,
        detect_lora_type,
        effective_rank,
        layer_type,
        parse_json_field,
        parse_lora_key,
        read_header,
        read_tensor_bytes,
    )
except ImportError:
    from lora_meta import (
        HAS_NUMPY,
        bytes_to_floats,
        compute_stats,
        detect_architecture,
        detect_lora_type,
        effective_rank,
        layer_type,
        parse_json_field,
        parse_lora_key,
        read_header,
        read_tensor_bytes,
    )

try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None


W = 72


def section(title):
    print(f"\n{'═' * W}\n  {title}\n{'═' * W}")


def sub(title):
    pad = W - 6 - len(title)
    print(f"\n  ── {title} {'─' * max(0, pad)}")


def row(label, value, w=36, indent=4):
    s = str(value)
    if len(s) > 100:
        s = s[:97] + "..."
    print(f"{' ' * indent}{label:<{w}} {s}")


def table_header(*cols, widths):
    header = "  " + "".join(f"{c:<{w}}" for c, w in zip(cols, widths))
    print(header)
    print("  " + "─" * (sum(widths)))


def table_row(*cols, widths):
    print("  " + "".join(f"{str(c):<{w}}" for c, w in zip(cols, widths)))


def analyse(path: Path):
    section(f"FORENSIC ANALYSIS: {path.name}")
    row("Path", path.resolve())
    row("Size", f"{path.stat().st_size / 1024 / 1024:.2f} MB")

    meta, header, data_offset = read_header(path)

    sub("Embedded Metadata")
    if meta:
        for key in sorted(meta.keys()):
            value = parse_json_field(meta[key])
            if isinstance(value, (dict, list)):
                print(f"    {key}:")
                for line in json.dumps(value, indent=6).splitlines():
                    print(f"      {line}")
            else:
                row(key, value)
    else:
        print("    (none)")

    all_keys = list(header.keys())
    arch = detect_architecture(all_keys)
    lora_type = detect_lora_type(all_keys)

    sub("Architecture Fingerprint")
    row("Architecture", arch)
    row("LoRA type", lora_type)
    row("Total tensors", len(all_keys))

    layers = defaultdict(dict)
    for key, info in header.items():
        base, role = parse_lora_key(key)
        layers[base][role] = info

    sub("Rank Analysis (from tensor shapes)")
    ranks = defaultdict(int)
    alpha_values = {}
    layer_ranks = {}

    for base, roles in layers.items():
        down = roles.get("lora_down") or roles.get("lora_A")
        alp = roles.get("alpha")

        if down:
            shape = down.get("shape", [])
            if shape:
                r = shape[0]
                ranks[r] += 1
                layer_ranks[base] = r

        if alp:
            try:
                raw = read_tensor_bytes(path, alp, data_offset)
                arr = bytes_to_floats(raw, alp.get("dtype", "F32"))
                if arr is not None and len(arr) > 0:
                    alpha_values[base] = float(arr[0])
            except Exception:
                pass

    if ranks:
        print(f"    {'Rank':<10} {'Layer count':>12}")
        print(f"    {'─'*10} {'─'*12}")
        for r, cnt in sorted(ranks.items()):
            print(f"    {r:<10} {cnt:>12}")
    else:
        print("    (could not determine rank from shapes)")

    if alpha_values:
        unique_alphas = sorted(set(round(v, 4) for v in alpha_values.values()))
        row("\nAlpha values found", unique_alphas)
        for base, alpha in list(alpha_values.items())[:3]:
            rank = layer_ranks.get(base)
            if rank:
                row(f"  scale ({base[:40]})", f"{alpha}/{rank} = {alpha/rank:.4f}")

    sub("Layer Coverage")
    type_counts = defaultdict(int)
    for base in layers:
        type_counts[layer_type(base)] += 1

    table_header("Layer type", "Count", widths=[30, 10])
    for lt, cnt in sorted(type_counts.items(), key=lambda item: -item[1]):
        table_row(lt, cnt, widths=[30, 10])

    sub("Per-Layer Shape Table (down / up / rank)")
    print(f"    {'Layer (truncated)':<52} {'down shape':<20} {'up shape':<20} {'rank':>6}")
    print(f"    {'─'*52} {'─'*20} {'─'*20} {'─'*6}")
    for base in sorted(layers.keys()):
        roles = layers[base]
        dn = roles.get("lora_down") or roles.get("lora_A")
        up = roles.get("lora_up") or roles.get("lora_B")
        dn_shape = str(dn["shape"]) if dn and "shape" in dn else "—"
        up_shape = str(up["shape"]) if up and "shape" in up else "—"
        rank = str(layer_ranks.get(base, "?"))
        label = base[-52:] if len(base) > 52 else base
        print(f"    {label:<52} {dn_shape:<20} {up_shape:<20} {rank:>6}")

    if HAS_NUMPY and np is not None:
        sub("Weight Statistics per Layer (requires numpy)")
        print(f"\n    Analysing {len(layers)} layers — this may take a moment...\n")

        grand_norms = []
        layer_stats = []
        for base in sorted(layers.keys()):
            roles = layers[base]
            dn_info = roles.get("lora_down") or roles.get("lora_A")
            up_info = roles.get("lora_up") or roles.get("lora_B")
            if not dn_info or not up_info:
                continue
            try:
                dn_raw = read_tensor_bytes(path, dn_info, data_offset)
                up_raw = read_tensor_bytes(path, up_info, data_offset)
                dn_arr = bytes_to_floats(dn_raw, dn_info.get("dtype", "F32"))
                up_arr = bytes_to_floats(up_raw, up_info.get("dtype", "F32"))
            except Exception as exc:
                layer_stats.append((base, None, None, str(exc)))
                continue

            dn_stats = compute_stats(dn_arr)
            up_stats = compute_stats(up_arr)
            try:
                dn_shape = dn_info["shape"]
                up_shape = up_info["shape"]
                dn_mat = dn_arr.reshape(dn_shape[0], -1).astype(np.float32)
                up_mat = up_arr.reshape(up_shape[0], -1).astype(np.float32)
                delta_w = up_mat @ dn_mat
                dw_norm = float(np.linalg.norm(delta_w, "fro"))
                dw_max = float(np.max(np.abs(delta_w)))
                eff_rank = effective_rank(dn_arr, dn_shape)
                alpha = alpha_values.get(base)
                rank_val = layer_ranks.get(base, 1)
                scale = (alpha / rank_val) if alpha else 1.0
                dw_scaled = dw_norm * scale
                grand_norms.append(dw_scaled)
                layer_stats.append((base, dn_stats, up_stats, {
                    "dw_frob_norm": round(dw_norm, 4),
                    "dw_scaled_norm": round(dw_scaled, 4),
                    "dw_max_weight": round(dw_max, 6),
                    "effective_rank": eff_rank,
                    "alpha": round(alpha, 4) if alpha else "n/a",
                    "scale": round(scale, 4),
                }))
            except Exception as exc:
                layer_stats.append((base, dn_stats, up_stats, f"(delta-W error: {exc})"))

        print(f"    {'Layer':<48} {'ΔW scaled‑norm':>16} {'eff‑rank':>10} {'α/rank':>8} {'↑ signal?':>10}")
        print(f"    {'─'*48} {'─'*16} {'─'*10} {'─'*8} {'─'*10}")

        median_norm = sorted(grand_norms)[len(grand_norms)//2] if grand_norms else 1.0
        for base, _, _, extra in layer_stats:
            label = base[-48:] if len(base) > 48 else base
            if isinstance(extra, dict):
                n = f"{extra['dw_scaled_norm']:.4f}"
                er = str(extra['effective_rank']) if extra['effective_rank'] else "n/a"
                sc = str(extra['scale'])
                sig = "  ★" if extra["dw_scaled_norm"] > median_norm * 1.5 else ""
                print(f"    {label:<48} {n:>16} {er:>10} {sc:>8} {sig:>10}")
            else:
                print(f"    {label:<48} {'error':>16} {'—':>10} {'—':>8} {'':>10}")
    else:
        sub("Weight Statistics")
        print("    ⚠  numpy not available — install it for full weight analysis.")
        print("       pip install numpy")


def main():
    cwd = Path(".")
    files = sorted(cwd.glob("*.safetensors"))
    if not files:
        print("No .safetensors files found in current directory.")
        sys.exit(1)
    print(f"Found {len(files)} .safetensors file(s) in {cwd.resolve()}")
    if not HAS_NUMPY:
        print("  ⚠  numpy not found — weight stats disabled. Run: pip install numpy")
    for file in files:
        analyse(file)
    print(f"{'═' * W}\n  Done. {len(files)} file(s).\n{'═' * W}\n")


if __name__ == "__main__":
    main()

