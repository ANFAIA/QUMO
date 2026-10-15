"""Writer / reader for QUMO's learned-model container.

Formats: ``qumo.rlsynth.v0`` (synthesis policies) and ``qumo.rlroute.v0``
(routing policies). A model is a directory ``<id>/`` with

* ``manifest.json``: everything except the weights (see ``write_model``);
* ``weights.bin``: raw little-endian tensors, each starting on a 64-byte
  boundary, dtype ``f32`` or ``f16`` per tensor (f16 is decoded to f32 by the
  loader). Integrity: ``fnv1a64`` (16 hex digits, FNV-1a 64 over the whole
  file, checked by the Mojo loader, QM-E617 on mismatch) and ``sha256``.

Tensor naming for the twisterl ``BasicPolicy`` MLP (arch.type = basic_mlp):

* ``embeddings.weight`` stored TRANSPOSED as [OBS, EMB] (arch.first_layer_transposed
  = true) so the sparse embedding-bag sums contiguous rows;
* ``embeddings.bias`` [EMB];
* ``common.<k>.weight`` [out, in], ``common.<k>.bias``;
* ``action.<k>.weight`` / ``.bias``; ``value.<k>.weight`` / ``.bias``.

The writer is used by training/ai_transpiler/synth (front 'policy') and
training/ai_transpiler/route (front 'router'). Pure numpy + stdlib.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from topology import fnv1a64, wl_hash  # noqa: E402  (same directory)

FORMATS = ("qumo.rlsynth.v0", "qumo.rlroute.v0")
ALIGN = 64


def _pad(blob: bytearray):
    while len(blob) % ALIGN:
        blob += b"\0"


def write_model(out_dir: str, manifest: dict, tensors, dtype: str = "f16", f32_names=()):
    """Write ``manifest.json`` + ``weights.bin`` into ``out_dir``.

    manifest: all descriptive fields (format, id, kind, num_qubits, edges,
      actions, obs, arch, symmetries, test_vectors, rollouts, training,
      license ...). The writer fills ``tensors``, ``weights_file``,
      ``weights_bytes``, ``fnv1a64`` and ``sha256``.
    tensors: ordered list of (name, np.ndarray). Arrays are stored row-major.
    dtype: 'f16' or 'f32' default storage; names in ``f32_names`` stay f32
      (biases are tiny, keep them exact).
    Returns (manifest, decoded) where decoded maps name -> float32 array exactly
    as the loader will see it (after f16 rounding), so callers compute test
    vectors from the stored weights.
    """
    if manifest.get("format") not in FORMATS:
        raise ValueError(f"format must be one of {FORMATS}")
    os.makedirs(out_dir, exist_ok=True)
    blob = bytearray()
    ents = []
    decoded = {}
    for name, arr in tensors:
        arr = np.ascontiguousarray(np.asarray(arr, dtype=np.float32))
        dt = "f32" if (dtype == "f32" or name in f32_names or name.endswith(".bias")) else "f16"
        _pad(blob)
        off = len(blob)
        if dt == "f16":
            data = arr.astype("<f2")
            if not np.all(np.isfinite(data)):
                raise ValueError(f"tensor {name} overflows f16")
            decoded[name] = data.astype(np.float32)
        else:
            data = arr.astype("<f4")
            decoded[name] = data.astype(np.float32)
        blob += data.tobytes(order="C")
        ents.append({"name": name, "shape": list(arr.shape), "dtype": dt, "offset": off})
    _pad(blob)
    raw = bytes(blob)
    with open(os.path.join(out_dir, "weights.bin"), "wb") as f:
        f.write(raw)
    m = dict(manifest)
    m["tensors"] = ents
    m["weights_file"] = "weights.bin"
    m["weights_bytes"] = len(raw)
    m["fnv1a64"] = f"{fnv1a64(raw):016x}"
    m["sha256"] = hashlib.sha256(raw).hexdigest()
    m.setdefault("license", "Apache-2.0")
    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        json.dump(m, f, separators=(",", ":"))
    # round-trip self-check
    m2, t2 = read_model(out_dir)
    for k, v in decoded.items():
        if not np.array_equal(t2[k], v):
            raise RuntimeError(f"round trip mismatch for {k}")
    return m, decoded


def read_model(model_dir: str):
    """Load (manifest, {name: float32 array}) and verify the FNV checksum."""
    with open(os.path.join(model_dir, "manifest.json")) as f:
        m = json.load(f)
    if m.get("format") not in FORMATS:
        raise ValueError(f"unknown format {m.get('format')!r}")
    with open(os.path.join(model_dir, m.get("weights_file", "weights.bin")), "rb") as f:
        raw = f.read()
    if f"{fnv1a64(raw):016x}" != m["fnv1a64"]:
        raise ValueError("fnv1a64 mismatch")
    out = {}
    for e in m["tensors"]:
        cnt = int(np.prod(e["shape"]))
        dt = "<f2" if e["dtype"] == "f16" else "<f4"
        out[e["name"]] = np.frombuffer(raw, dtype=dt, count=cnt, offset=e["offset"]).astype(np.float32).reshape(e["shape"])
    return m, out


def file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def update_index(root: str):
    """Rebuild ``<root>/INDEX.json`` from every ``<root>/<id>/manifest.json``.

    Entry: {id, kind, num_qubits, edges, wl_hash, dir, sha256, bytes}
    (sha256 of weights.bin, bytes = weights.bin + manifest.json size).
    """
    entries = []
    for d in sorted(os.listdir(root)):
        mp = os.path.join(root, d, "manifest.json")
        if not os.path.isfile(mp):
            continue
        with open(mp) as f:
            m = json.load(f)
        wp = os.path.join(root, d, m.get("weights_file", "weights.bin"))
        edges = [list(e) for e in m.get("edges", [])]
        n = int(m.get("num_qubits", 0))
        entries.append({
            "id": m["id"], "kind": m["kind"], "num_qubits": n, "edges": edges,
            "wl_hash": wl_hash(n, [tuple(e) for e in edges]) if n > 0 else "",
            "dir": d, "sha256": file_sha256(wp),
            "bytes": os.path.getsize(wp) + os.path.getsize(mp),
        })
    with open(os.path.join(root, "INDEX.json"), "w") as f:
        json.dump(entries, f, indent=1)
    return entries


def basic_mlp_tensors(state_dict):
    """Order + transpose a twisterl BasicPolicy state_dict (torch or numpy values)
    into the container's tensor list."""
    def arr(v):
        return v.detach().cpu().float().numpy() if hasattr(v, "detach") else np.asarray(v, dtype=np.float32)
    sd = {k: arr(v) for k, v in state_dict.items()}
    out = [("embeddings.weight", np.ascontiguousarray(sd["embeddings.weight"].T)),
           ("embeddings.bias", sd["embeddings.bias"])]
    for prefix in ("common", "action", "value"):
        idx = sorted({int(k.split(".")[1]) for k in sd if k.startswith(prefix + ".")})
        for j in idx:
            out.append((f"{prefix}.{j}.weight", sd[f"{prefix}.{j}.weight"]))
            out.append((f"{prefix}.{j}.bias", sd[f"{prefix}.{j}.bias"]))
    return out
