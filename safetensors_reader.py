"""
Minimal safetensors reader helpers reused by LoRA analysis code.
"""

from __future__ import annotations

import json
import struct
from pathlib import Path

try:  # pragma: no cover - optional dependency
    import numpy as np
    HAS_NUMPY = True
except ImportError:  # pragma: no cover
    HAS_NUMPY = False


def read_header(path: Path):
    with open(path, "rb") as f:
        raw_len = f.read(8)
        if len(raw_len) != 8:
            raise ValueError(f"Invalid safetensors file header for {path}: missing length prefix")
        header_len = struct.unpack("<Q", raw_len)[0]
        file_size = path.stat().st_size
        if header_len <= 0:
            raise ValueError(f"Invalid safetensors header size for {path}: {header_len}")
        if header_len > file_size - 8:
            raise ValueError(
                f"Invalid safetensors header size for {path}: {header_len} exceeds file payload"
            )
        raw_header = f.read(header_len)
        if len(raw_header) != header_len:
            raise ValueError(f"Incomplete safetensors header read for {path}")
        data_offset = 8 + header_len
    header = json.loads(raw_header.decode("utf-8"))
    meta = header.pop("__metadata__", {})
    return meta, header, data_offset


def read_tensor_bytes(path: Path, info: dict, data_offset: int) -> bytes:
    offsets = info.get("data_offsets", [0, 0])
    start, end = offsets[0], offsets[1]
    with open(path, "rb") as f:
        f.seek(data_offset + start)
        return f.read(end - start)


def bytes_to_floats(raw: bytes, dtype: str):
    if not HAS_NUMPY:
        return None
    dt_map = {
        "F32": np.float32, "F64": np.float64,
        "F16": np.float16, "BF16": "bfloat16",
        "I32": np.int32, "I16": np.int16, "I8": np.int8,
    }
    if dtype == "BF16":
        arr16 = np.frombuffer(raw, dtype=np.uint16)
        arr32 = arr16.astype(np.uint32) << 16
        return arr32.view(np.float32)
    dt = dt_map.get(dtype)
    if dt is None:
        return None
    return np.frombuffer(raw, dtype=dt)


def parse_json_field(v):
    if isinstance(v, str):
        try:
            return json.loads(v)
        except Exception:
            pass
    return v

