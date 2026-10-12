#!/usr/bin/env python3
"""QUMO unified benchmark runner: CPU, GPU (NVIDIA / AMD / Apple) and baselines.

    python bench/run_benchmarks.py --list
    python bench/run_benchmarks.py --config bench/configs/quick.json
    python bench/run_benchmarks.py --config auto --out bench/results/cloud/
    python bench/run_benchmarks.py --config colab_t4 --suites sv_gpu,baseline_cudaq --quick
    python bench/run_benchmarks.py --config auto --build-only     # compile everything, run nothing
    python bench/run_benchmarks.py --select-config auto            # detect the GPU, print its preset
    python bench/run_benchmarks.py --select-config "NVIDIA L4"     # print the preset for a GPU name
    python bench/run_benchmarks.py --select-config "" --gpu-vendor amd --gpu-arch gfx942 --gpu-memory-gib 192

A run executes the suites enabled in a JSON config (bench/configs/*.json) one
after the other and writes ONE result JSON (environment metadata, per-suite
status, commands, timing rows) plus a CSV with the rows. The JSON is rewritten
after every suite, so an interrupted cloud session keeps what finished.

Suites (each optional; failures and timeouts are recorded, never fatal):

    gpu_check          make gpu-check for the host GPU arch (+ any listed targets)
    sv_cpu             QUMO CPU state vector, build/bench_sv (layered / qft / random)
    sv_gpu             QUMO GPU state vector + QEC Pauli frames, build/gpu/bench_gpu_{fp32,fp64}
    baseline_aer       Qiskit Aer statevector, CPU and (qiskit-aer-gpu) GPU, fp32 / fp64
    baseline_cudaq     NVIDIA CUDA-Q, target nvidia (fp32 / fp64), optional qpp-cpu
    baseline_lightning PennyLane Lightning: lightning.qubit / .gpu (NVIDIA) / .amdgpu (AMD) / .kokkos
    neutral            small subset of the hardware-neutral compiler benchmark (bench/neutral)
    qec                small subset of bench/bench_qec.py (QUMO vs Stim + PyMatching)
    tests              make test / make test-gpu (or a list of test files): pass counts

Every engine runs the SAME layered circuit as bench/bench_sv.mojo (10 layers of
ry+rz on every qubit, then a CX brick), on the same qubit grid (step 2), with a
warm-up and the median over `reps` repetitions. Qubit ranges are clamped to
60 % (configurable) of the GPU memory and of the free host RAM; every clamp is
recorded. Vendors are treated alike: there is no default vendor, presets are
named by hardware, and each vendor gets the strongest baseline that runs on it.

The runner itself is stdlib-only (psutil used when present). Baselines run in
worker subprocesses (`--worker <name>`) with the interpreter named in the config
or in QUMO_BENCH_PYTHON / QUMO_AER_GPU_PYTHON / QUMO_CUDAQ_PYTHON /
QUMO_LIGHTNING_PYTHON, so e.g. qiskit-aer-gpu (needs qiskit<2) and CUDA-Q live
in their own virtual environments. Error codes QM-E960..QM-E969 (docs/ERRORS.md).
"""

import argparse
import copy
import csv
import datetime as _dt
import glob
import json
import math
import os
import platform
import re
import shutil
import signal
import statistics
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_DIR = os.path.join(ROOT, "bench", "configs")
SCHEMA = "qumo-bench/1"
WORKER_TAG = "@@QUMO "

# Stable error codes of the benchmark harness (QM-E960..QM-E969).
E_CONFIG = "QM-E960"     # bad / unknown config, suite or argument
E_BUILD = "QM-E961"      # Mojo build of a benchmark binary failed
E_TIMEOUT = "QM-E962"    # a suite or command exceeded its timeout
E_TOOL = "QM-E963"       # required tool or binary missing (mojo, make, nvidia-smi ...)
E_PARSE = "QM-E964"      # benchmark output could not be parsed
E_BASELINE = "QM-E965"   # third-party baseline not installed / unusable
E_MEMORY = "QM-E966"     # requested qubit range does not fit the memory budget
E_GPU = "QM-E967"        # no usable GPU / unsupported GPU / precision unsupported
E_WORKER = "QM-E968"     # benchmark binary or baseline worker crashed / produced no result
E_FAIL = "QM-E969"       # a test or check reported FAIL

HINTS = {
    E_CONFIG: "run `python bench/run_benchmarks.py --list` for suites and presets",
    E_BUILD: "run the printed build command by hand; see docs/GPU.md for GPU toolchains",
    E_TIMEOUT: "raise `timeout_s` in the config or shrink the qubit range (--quick)",
    E_TOOL: "run `make setup` (Mojo toolchain) or install the missing tool",
    E_PARSE: "the benchmark binary output changed; check the raw `output_tail` in the JSON",
    E_BASELINE: "install the baseline in the interpreter given by the config (see docs/BENCHMARKING_ON_CLOUD_GPUS.md)",
    E_MEMORY: "use a larger GPU / high-RAM runtime or lower `qubits` in the config",
    E_GPU: "check `nvidia-smi` / `amd-smi` / `rocm-smi`; unsupported GPUs fall back to CPU + baselines",
    E_WORKER: "rerun the command recorded in `commands` to see the full error (MODULAR_DEBUG=device-sync-mode for GPU errors)",
    E_FAIL: "rerun the failing test by hand: .venv/bin/mojo run -I . <file>",
}


def err(code, msg):
    return f"{code}: {str(msg).rstrip('. ')}. Hint: {HINTS[code]}."


class BenchError(Exception):
    pass


# ----------------------------------------------------------------------------- small utilities

def now_iso():
    return _dt.datetime.now(_dt.timezone.utc).replace(microsecond=0).isoformat()


def loadavg():
    try:
        return [round(x, 2) for x in os.getloadavg()]
    except (AttributeError, OSError):
        return None


def median(xs):
    """Upper median, sorted(xs)[len // 2]: the rule of bench_sv.mojo / bench_gpu.mojo,
    so every engine's repetitions are reduced the same way (identical for odd counts)."""
    return sorted(xs)[len(xs) // 2] if xs else None


def tail(text, n=40):
    lines = [ln for ln in (text or "").splitlines() if "Crashpad" not in ln]
    return "\n".join(lines[-n:])


def run_cmd(cmd, timeout=None, cwd=ROOT, env=None):
    """Run a command in its own process group; kill the whole group on timeout.

    Returns {cmd, rc, stdout, stderr, seconds, timed_out}. Never raises for a
    failing command (rc != 0); raises BenchError only when the executable is
    missing."""
    t0 = time.perf_counter()
    try:
        p = subprocess.Popen(cmd, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                             text=True, errors="replace", start_new_session=True)
    except FileNotFoundError as e:
        raise BenchError(err(E_TOOL, f"cannot execute {cmd[0]!r} ({e})"))
    timed_out = False
    try:
        out, errs = p.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        timed_out = True
        try:
            os.killpg(p.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            p.kill()
        out, errs = p.communicate()
    return {"cmd": " ".join(map(str, cmd)), "rc": p.returncode, "stdout": out or "", "stderr": errs or "",
            "seconds": round(time.perf_counter() - t0, 3), "timed_out": timed_out}


def first_error_line(text):
    """Most informative error line of a crashed binary (QUMO code > exception > error)."""
    lines = [ln.strip() for ln in (text or "").splitlines() if ln.strip() and "Crashpad" not in ln]
    for pat in (r"QM-E\d{3}", r"Unhandled exception", r"(?i)\berror\b(?!.*stack-trace)", r"(?i)exception"):
        for ln in lines:
            if re.search(pat, ln) and "stack trace was not collected" not in ln:
                return ln[:400]
    return None


def which(name):
    return shutil.which(name)


def human_bytes(b):
    if b is None:
        return "?"
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(b) < 1024 or unit == "TiB":
            return f"{b:.1f} {unit}"
        b /= 1024.0


# ----------------------------------------------------------------------------- output parsing
# The Mojo benchmarks print CSV with " , " separators, '#' comment lines and
# possibly several tables (new header = new table). ALL parsing of their
# output goes through this one function, so format tweaks stay local.

def parse_bench_csv(text):
    """-> {"comments": [str], "tables": [{"header": [str], "rows": [dict]}]}.

    A header is a line whose comma-separated fields all start with a letter or
    '_' and contain no spaces; any other line with the header's field count is a
    data row. Numbers become int/float; other fields stay strings."""
    comments, tables = [], []
    cur = None
    for raw in (text or "").splitlines():
        line = raw.strip()
        if not line or "Crashpad" in line:
            continue
        if line.startswith("#"):
            comments.append(line.lstrip("#").strip())
            continue
        fields = [f.strip() for f in line.split(",")]
        if len(fields) < 2:
            continue
        if all(re.match(r"^[A-Za-z_][A-Za-z0-9_()./-]*$", f) for f in fields) and not all(_is_num(f) for f in fields):
            # A header has at least one field that is a known column word; data
            # rows such as "surface_z,3,3,..." contain numbers, so they never match.
            cur = {"header": [f.lower() for f in fields], "rows": []}
            tables.append(cur)
            continue
        if cur is None or len(fields) != len(cur["header"]):
            continue
        cur["rows"].append({k: _num(v) for k, v in zip(cur["header"], fields)})
    return {"comments": comments, "tables": tables}


def _is_num(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


def _num(s):
    try:
        return int(s)
    except ValueError:
        pass
    try:
        return float(s)
    except ValueError:
        low = s.lower()
        if low in ("true", "false"):
            return low == "true"
        return s


def find_table(parsed, *required):
    for t in parsed["tables"]:
        if all(r in t["header"] for r in required):
            return t
    return None


def parse_markdown_tables(text):
    """Markdown tables (| a | b |) -> list of {"header", "rows"} (used for bench_qec.py)."""
    tables, cur = [], None
    for line in (text or "").splitlines():
        s = line.strip()
        if not (s.startswith("|") and s.endswith("|")):
            cur = None
            continue
        cells = [c.strip() for c in s.strip("|").split("|")]
        if cur is None:
            cur = {"header": cells, "rows": []}
            tables.append(cur)
        elif all(re.match(r"^:?-{2,}:?$", c) for c in cells if c):
            continue
        elif len(cells) == len(cur["header"]):
            cur["rows"].append({k: _num(v) for k, v in zip(cur["header"], cells)})
    return tables


# ----------------------------------------------------------------------------- environment

def _read(path):
    try:
        with open(path) as fh:
            return fh.read()
    except OSError:
        return None


def host_memory():
    """-> (total_bytes, available_bytes)."""
    try:
        import psutil  # optional
        vm = psutil.virtual_memory()
        return int(vm.total), int(vm.available)
    except Exception:
        pass
    info = _read("/proc/meminfo")
    if info:
        def kb(key):
            m = re.search(rf"^{key}:\s+(\d+)\s+kB", info, re.M)
            return int(m.group(1)) * 1024 if m else None
        total = kb("MemTotal")
        return total, kb("MemAvailable") or total
    if sys.platform == "darwin":
        r = subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True)
        total = int(r.stdout.strip()) if r.returncode == 0 else None
        return total, int(total * 0.7) if total else None  # no cheap "available" on macOS
    return None, None


def cpu_info():
    info = {"logical_cores": os.cpu_count(), "machine": platform.machine(), "model": platform.processor() or None}
    try:
        info["usable_cores"] = len(os.sched_getaffinity(0))
    except AttributeError:
        info["usable_cores"] = os.cpu_count()
    if sys.platform == "darwin":
        for key, name in (("machdep.cpu.brand_string", "model"), ("hw.physicalcpu", "physical_cores")):
            r = subprocess.run(["sysctl", "-n", key], capture_output=True, text=True)
            if r.returncode == 0:
                v = r.stdout.strip()
                info[name] = int(v) if v.isdigit() else v
    else:
        txt = _read("/proc/cpuinfo") or ""
        m = re.search(r"^model name\s*:\s*(.+)$", txt, re.M)
        if m:
            info["model"] = m.group(1).strip()
        flags = re.search(r"^flags\s*:\s*(.+)$", txt, re.M)
        if flags:
            fl = set(flags.group(1).split())
            # x86-64-v3 is the MAX / Mojo CPU baseline (docs.modular.com/packages)
            info["x86_64_v3"] = all(f in fl for f in ("avx2", "bmi1", "bmi2", "fma", "f16c", "movbe"))
    return info


def os_info():
    d = {"system": platform.system(), "release": platform.release(), "platform": platform.platform()}
    osr = _read("/etc/os-release")
    if osr:
        m = re.search(r'^PRETTY_NAME="?([^"\n]+)"?', osr, re.M)
        if m:
            d["distro"] = m.group(1)
    libc = platform.libc_ver()
    if libc and libc[0]:
        d["libc"] = "-".join(libc)  # MAX wheels need glibc >= 2.34
    if sys.platform == "darwin":
        d["macos"] = platform.mac_ver()[0]
    return d


def detect_nvidia():
    if not which("nvidia-smi"):
        return []
    q = "index,name,memory.total,driver_version,compute_cap,pci.bus_id"
    r = subprocess.run(["nvidia-smi", f"--query-gpu={q}", "--format=csv,noheader,nounits"],
                       capture_output=True, text=True)
    if r.returncode != 0:  # very old drivers: no compute_cap field
        q = "index,name,memory.total,driver_version"
        r = subprocess.run(["nvidia-smi", f"--query-gpu={q}", "--format=csv,noheader,nounits"],
                           capture_output=True, text=True)
        if r.returncode != 0:
            return []
    gpus = []
    for line in r.stdout.strip().splitlines():
        f = [x.strip() for x in line.split(",")]
        g = {"vendor": "nvidia", "index": _num(f[0]), "name": f[1], "source": "nvidia-smi"}
        try:
            g["memory_bytes"] = int(float(f[2])) * 1024 * 1024
        except (ValueError, IndexError):
            pass
        if len(f) > 3:
            g["driver"] = f[3]
        if len(f) > 4 and re.match(r"^\d+\.\d+$", f[4]):
            major, minor = f[4].split(".")
            g["compute_capability"] = f[4]
            g["arch"] = f"sm_{major}{minor}"
        gpus.append(g)
    r2 = subprocess.run(["nvidia-smi"], capture_output=True, text=True)
    m = re.search(r"CUDA Version:\s*([\d.]+)", r2.stdout or "")
    for g in gpus:
        if m:
            g["cuda_driver_api"] = m.group(1)
    return gpus


def _walk(obj):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield k, v
            yield from _walk(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from _walk(v)


def detect_amd():
    gpus = []
    if which("amd-smi"):
        r = subprocess.run(["amd-smi", "static", "--json"], capture_output=True, text=True)
        try:
            data = json.loads(r.stdout)
            items = data if isinstance(data, list) else data.get("gpu_data", [data]) if isinstance(data, dict) else []
            for i, item in enumerate(items):
                g = {"vendor": "amd", "index": i, "source": "amd-smi"}
                for k, v in _walk(item):
                    if k == "market_name" and isinstance(v, str):
                        g["name"] = v
                    elif k == "target_graphics_version" and isinstance(v, str):
                        g["arch"] = v
                    elif k == "vram" and isinstance(v, dict):
                        size = v.get("size")
                        if isinstance(size, dict) and _is_num(str(size.get("value", ""))):
                            unit = str(size.get("unit", "MB")).upper()
                            mult = {"B": 1, "KB": 1024, "MB": 1024 ** 2, "GB": 1024 ** 3}.get(unit, 1024 ** 2)
                            g["memory_bytes"] = int(float(size["value"]) * mult)
                    elif k in ("driver_version", "version") and isinstance(v, str) and "driver" not in g:
                        g["driver"] = v
                if g.get("name") or g.get("arch"):
                    gpus.append(g)
        except (ValueError, AttributeError, TypeError):
            pass
    if not gpus and which("rocm-smi"):
        r = subprocess.run(["rocm-smi", "--showproductname", "--showmeminfo", "vram", "--json"],
                           capture_output=True, text=True)
        try:
            data = json.loads(r.stdout)
            for i, (card, d) in enumerate(sorted(data.items())):
                if not card.startswith("card"):
                    continue
                g = {"vendor": "amd", "index": i, "source": "rocm-smi"}
                for k, v in d.items():
                    kl = k.lower()
                    if ("card series" in kl or "card sku" in kl) and "name" not in g:
                        g["name"] = v
                    if "gfx version" in kl:
                        g["arch"] = v
                    if "vram total memory" in kl and _is_num(str(v)):
                        g["memory_bytes"] = int(v)
                gpus.append(g)
        except (ValueError, AttributeError):
            pass
    if which("rocminfo"):
        r = subprocess.run(["rocminfo"], capture_output=True, text=True)
        agents = re.findall(r"Name:\s+(gfx\w+)\s*\n.*?Marketing Name:\s+(.+?)\s*\n", r.stdout or "", re.S)
        if not gpus:
            for i, (arch, name) in enumerate(agents):
                gpus.append({"vendor": "amd", "index": i, "name": name, "arch": arch, "source": "rocminfo"})
        else:
            for g, (arch, name) in zip(gpus, agents):
                g.setdefault("arch", arch)
                g.setdefault("name", name)
    rocm_ver = (_read("/opt/rocm/.info/version") or "").strip() or None
    for g in gpus:
        if rocm_ver:
            g["rocm"] = rocm_ver
        if "memory_bytes" not in g:
            for p in sorted(glob.glob("/sys/class/drm/card*/device/mem_info_vram_total")):
                v = (_read(p) or "").strip()
                if v.isdigit():
                    g["memory_bytes"] = int(v)
                    break
    return gpus


def detect_apple():
    if sys.platform != "darwin" or platform.machine() != "arm64":
        return []
    g = {"vendor": "apple", "index": 0, "source": "system_profiler", "arch": "metal:4"}
    r = subprocess.run(["system_profiler", "SPDisplaysDataType", "-json"], capture_output=True, text=True)
    try:
        d = json.loads(r.stdout)["SPDisplaysDataType"][0]
        g["name"] = d.get("sppci_model") or d.get("_name")
        if d.get("sppci_cores"):
            g["gpu_cores"] = _num(str(d["sppci_cores"]))
        if d.get("spdisplays_mtlgpufamilysupport"):
            g["metal_family"] = d["spdisplays_mtlgpufamilysupport"].replace("spdisplays_", "")
    except (ValueError, KeyError, IndexError):
        g["name"] = "Apple GPU"
    total, _ = host_memory()
    if total:
        # Unified memory; Metal's recommendedMaxWorkingSetSize is ~75 % of RAM.
        g["memory_bytes"] = int(total * 0.75)
        g["memory_note"] = "unified memory: 75% of host RAM (Metal working-set limit)"
    r2 = subprocess.run(["xcrun", "-f", "metal"], capture_output=True, text=True)
    g["metal_toolchain"] = r2.returncode == 0
    return [g]


def detect_gpus():
    gpus = detect_nvidia()
    gpus += detect_amd()
    gpus += detect_apple()
    return gpus


def mojo_versions(root, mojo):
    out = {"mojo_path": os.path.relpath(mojo, root) if mojo and mojo.startswith(root) else mojo}
    if mojo and os.path.exists(mojo):
        r = subprocess.run([mojo, "--version"], capture_output=True, text=True)
        out["mojo"] = tail(r.stdout, 1).strip() or None
    for dist in ("mojo", "mojo_compiler", "max_core", "max_mojo_libs", "max"):
        hits = glob.glob(os.path.join(root, ".venv", "lib", "python*", "site-packages", f"{dist}-*.dist-info"))
        if hits:
            out[dist.replace("_", "-")] = os.path.basename(hits[0])[len(dist) + 1:-len(".dist-info")]
    return out


def _sanitize_url(url):
    return re.sub(r"(https?://)[^@/]+@", r"\1", url or "")  # never record credentials


def git_info(root):
    def g(*args):
        r = subprocess.run(["git", "-C", root, *args], capture_output=True, text=True)
        return r.stdout.strip() if r.returncode == 0 else None
    if not which("git"):
        return {}
    status = g("status", "--porcelain")
    return {"commit": g("rev-parse", "HEAD"), "branch": g("rev-parse", "--abbrev-ref", "HEAD"),
            "describe": g("describe", "--always", "--dirty"),
            "dirty_files": len(status.splitlines()) if status else 0,
            "remote": _sanitize_url(g("remote", "get-url", "origin"))}


def cloud_info():
    d = {}
    if os.environ.get("COLAB_RELEASE_TAG") or os.environ.get("COLAB_GPU") is not None:
        d["provider"] = "google-colab"
        d["colab_release"] = os.environ.get("COLAB_RELEASE_TAG")
    elif os.path.exists("/opt/rocm") and os.environ.get("JUPYTERHUB_USER"):
        d["provider"] = "jupyterhub-rocm"
    keys = ("CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES", "OMP_NUM_THREADS")
    d["env"] = {k: os.environ[k] for k in keys if k in os.environ}
    d["env"]["MODULAR_NVPTX_COMPILER_PATH_set"] = bool(os.environ.get("MODULAR_NVPTX_COMPILER_PATH"))
    return d


def collect_env(root, mojo):
    total, avail = host_memory()
    gpus = detect_gpus()
    for g in gpus:
        g["mojo_max_supported"], g["support_note"] = gpu_supported(g)
    return {
        "timestamp_utc": now_iso(),
        "loadavg_start": loadavg(),
        "os": os_info(),
        "cpu": cpu_info(),
        "ram": {"total_bytes": total, "available_bytes": avail},
        "python": {"version": platform.python_version(), "implementation": platform.python_implementation()},
        "toolchain": mojo_versions(root, mojo),
        "git": git_info(root),
        "gpus": gpus,
        "cloud": cloud_info(),
    }


# ----------------------------------------------------------------------------- config selection

PRESETS_BY_NAME = [  # (regex on the lower-cased GPU name, preset); first match wins
    (r"\bt4\b", "colab_t4"),
    (r"\bl4\b", "colab_l4"),
    (r"\ba100\b", "colab_a100"),
    (r"\b(gh200|h100|h200|h800)\b", "colab_h100"),
    (r"\b(b100|b200|b300|gb200)\b", "colab_h100"),
    (r"rtx pro 6000", "colab_g4"),
    (r"\bmi3(00|25|50|55)x?\b", "amd_mi300x"),
]
# Big CDNA3/4 parts by architecture when the marketing name is missing
# (some rocm-smi versions print only a PCI id): >= 64 GiB of HBM.
PRESETS_BY_ARCH = [(r"^gfx9(42|50)", 64, "amd_mi300x")]
VENDOR_HINTS = {  # name fragments -> vendor, used only when the vendor is not given
    "nvidia": ("nvidia", "tesla", "geforce", "quadro", "rtx"),
    "amd": ("amd", "radeon", "instinct"),
    "apple": ("apple",),
}


# GPUs that Mojo / MAX 26.6 support out of the box (mojolang.org/docs/requirements,
# Oct 2026). NVIDIA: Turing (sm_75) and newer with driver >= 580 (older drivers
# work with MODULAR_NVPTX_COMPILER_PATH=<CUDA ptxas>). AMD: CDNA2-4, RDNA2-4.
AMD_SUPPORTED = {"gfx90a", "gfx942", "gfx950", "gfx1030", "gfx1033", "gfx1100", "gfx1101", "gfx1102", "gfx1103",
                 "gfx1150", "gfx1151", "gfx1152", "gfx1200", "gfx1201"}


def gpu_supported(gpu):
    """-> (supported, note) for a detected GPU dict (vendor/arch/driver)."""
    if not gpu:
        return False, "no GPU"
    v, arch = gpu.get("vendor"), (gpu.get("arch") or "")
    if v == "nvidia":
        m = re.match(r"sm_(\d+)", arch)
        if m and int(m.group(1)) < 75:
            return False, f"{arch} is pre-Turing; Mojo/MAX support starts at sm_75 (T4)"
        drv = gpu.get("driver") or ""
        major = int(drv.split(".")[0]) if drv[:1].isdigit() else None
        if major is not None and major < 580 and not os.environ.get("MODULAR_NVPTX_COMPILER_PATH"):
            return True, (f"driver {drv} < 580: set MODULAR_NVPTX_COMPILER_PATH to a CUDA ptxas "
                          "(e.g. /usr/local/cuda/bin/ptxas) if kernels fail to load")
        return True, "" if m else "compute capability unknown"
    if v == "amd":
        if arch and arch.split(":")[0] not in AMD_SUPPORTED:
            return False, f"{arch} is not in the Mojo/MAX AMD support list"
        return True, ""
    if v == "apple":
        return True, "" if gpu.get("metal_toolchain", True) else "Metal toolchain missing"
    return False, f"unknown vendor {v}"


def guess_vendor(gpu_name):
    name = (gpu_name or "").lower()
    for vendor, keys in VENDOR_HINTS.items():
        if any(k in name for k in keys):
            return vendor
    return "amd" if re.search(r"\bmi\d{3}", name) else ""


def select_config(gpu_name=None, vendor=None, memory_gib=None, arch=None):
    """Preset name for a GPU. Mirrors pick_config() in the example notebooks.

    The same rule for every vendor: a known model name (or, for AMD CDNA3/4,
    the architecture) selects its preset; any other GPU of a known vendor gets
    that vendor's generic preset (`nvidia_generic`, `amd_generic`,
    `apple_metal`), whose widths the runtime memory clamp then fits to the
    actual card. An unknown vendor or no GPU gives `cpu_only`: there is no
    default vendor."""
    name = (gpu_name or "").lower()
    vendor = (vendor or "").lower() or guess_vendor(name)
    for pat, preset in PRESETS_BY_NAME:
        if re.search(pat, name):
            return preset
    for pat, min_gib, preset in PRESETS_BY_ARCH:
        if arch and re.search(pat, arch.lower()) and (memory_gib or 0) >= min_gib:
            return preset
    return {"nvidia": "nvidia_generic", "amd": "amd_generic", "apple": "apple_metal"}.get(vendor, "cpu_only")


def auto_config(gpus):
    """Preset for the first detected GPU (what `--config auto` uses)."""
    g = gpus[0] if gpus else {}
    return select_config(g.get("name"), g.get("vendor"), (g.get("memory_bytes") or 0) / 2 ** 30, g.get("arch"))


def resolve_config_path(spec, gpus):
    if spec in (None, "", "auto"):
        spec = auto_config(gpus)
    if os.path.exists(spec):
        return spec
    cand = os.path.join(CONFIG_DIR, spec if spec.endswith(".json") else spec + ".json")
    if os.path.exists(cand):
        return cand
    raise BenchError(err(E_CONFIG, f"config {spec!r} not found (looked in {os.path.relpath(CONFIG_DIR, ROOT)})"))


def load_config(path):
    try:
        with open(path) as fh:
            cfg = json.load(fh)
    except ValueError as e:
        raise BenchError(err(E_CONFIG, f"{path} is not valid JSON ({e})"))
    if not isinstance(cfg.get("suites"), dict):
        raise BenchError(err(E_CONFIG, f"{path} has no 'suites' object"))
    unknown = [s for s in cfg["suites"] if s not in SUITES]
    if unknown:
        raise BenchError(err(E_CONFIG, f"{path}: unknown suite(s) {unknown}; known: {list(SUITES)}"))
    return cfg


def apply_quick(scfg):
    """--quick: use the suite's own "quick" overrides, else shrink generically."""
    s = copy.deepcopy(scfg)
    if isinstance(s.get("quick"), dict):
        s.update(s["quick"])
        return s
    for key in list(s):
        if key.startswith("qubits") and isinstance(s[key], list) and len(s[key]) == 2:
            lo, hi = s[key]
            s[key] = [lo, min(hi, lo + 4)]
    if "reps" in s:
        s["reps"] = 1
    return s


# ----------------------------------------------------------------------------- run context

class Ctx:
    def __init__(self, args, cfg, env, out_json):
        self.args = args
        self.cfg = cfg
        self.env = env
        self.out_json = out_json
        self.mojo = args.mojo
        self.build = not args.no_build
        self.mem_fraction = float(cfg.get("memory_fraction", 0.6))
        self.gpus = env["gpus"]
        self.gpu = self.gpus[0] if self.gpus else None
        self.vendor = self.gpu["vendor"] if self.gpu else "none"

    def log(self, msg):
        print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

    def host_budget(self):
        total, avail = host_memory()
        base = avail or total or 0
        return int(base * self.mem_fraction)

    def gpu_budget(self):
        if not self.gpu or not self.gpu.get("memory_bytes"):
            return None
        return int(self.gpu["memory_bytes"] * self.mem_fraction)

    def python_for(self, scfg, key="python", env_var=None):
        v = scfg.get(key)
        if env_var and os.environ.get(env_var):
            v = os.environ[env_var]
        if not v:
            v = os.environ.get("QUMO_BENCH_PYTHON") or sys.executable
        if not os.path.isabs(v) and os.sep in v:
            v = os.path.join(ROOT, v)
        return v


def qubit_grid(lo, hi, step=2):
    return list(range(int(lo), int(hi) + 1, int(step)))


def fit_qubits(grid, limits):
    """Drop widths whose memory need exceeds a budget.

    limits = [(label, budget_bytes or None, bytes_per_amplitude)]. Returns
    (kept_grid, clamp_record or None)."""
    kept, why = [], []
    for n in grid:
        ok = True
        for label, budget, bpa in limits:
            if budget is not None and bpa * (2 ** n) > budget:
                ok = False
                why.append(f"{n}q needs {human_bytes(bpa * 2 ** n)} > {label} budget {human_bytes(budget)}")
                break
        if ok:
            kept.append(n)
    clamp = None
    if len(kept) != len(grid):
        clamp = {"requested": grid, "used": kept, "reason": why[0] if why else "", "code": E_MEMORY}
    return kept, clamp


def needs_build(binary, sources):
    if not os.path.exists(binary):
        return True
    t = os.path.getmtime(binary)
    for src in sources:
        for path in ([src] if os.path.isfile(src) else glob.glob(os.path.join(src, "**", "*.mojo"), recursive=True)):
            if os.path.getmtime(path) > t:
                return True
    return False


def mojo_build(ctx, out, src, defines=(), timeout=1800, extra=()):
    """Build `src` into `out` unless up to date. -> (ok, info dict)."""
    srcs = [os.path.join(ROOT, src), os.path.join(ROOT, "qumo")]
    if not needs_build(out, srcs):
        return True, {"binary": os.path.relpath(out, ROOT), "built": False}
    if not ctx.build:  # a missing binary is a skip (recorded), not a failure
        return False, {"skip": True,
                       "error": err(E_TOOL, f"{os.path.relpath(out, ROOT)} missing or older than its sources "
                                            "and --no-build given")}
    if not ctx.mojo or not os.path.exists(ctx.mojo):
        return False, {"error": err(E_TOOL, f"Mojo compiler not found at {ctx.mojo}")}
    os.makedirs(os.path.dirname(out), exist_ok=True)
    cmd = [ctx.mojo, "build", "-I", ".", *sum((["-D", d] for d in defines), []), *extra, "-o", out, src]
    ctx.log("build: " + " ".join(cmd[1:]))
    r = run_cmd(cmd, timeout=timeout)
    info = {"binary": os.path.relpath(out, ROOT), "built": True, "command": r["cmd"].replace(ROOT + os.sep, ""),
            "seconds": r["seconds"]}
    if r["timed_out"]:
        info["error"] = err(E_TIMEOUT, f"build exceeded {timeout} s")
        return False, info
    if r["rc"] != 0 or not os.path.exists(out):
        info["error"] = err(E_BUILD, f"`{info['command']}` exited with {r['rc']}")
        info["output_tail"] = tail(r["stdout"] + "\n" + r["stderr"], 25)
        return False, info
    return True, info


def result(status, rows=None, reason=None, **kw):
    d = {"status": status, "rows": rows or []}
    if reason:
        d["reason"] = reason
    d.update(kw)
    return d


VENDOR_NAMES = {"nvidia": "NVIDIA", "amd": "AMD", "apple": "Apple", "none": "none"}


def label_for(engine, device, precision, extra=""):
    names = {"qumo": "QUMO", "aer": "Aer", "cudaq": "CUDA-Q", "lightning": "Lightning"}
    s = f"{names.get(engine, engine)} {device.upper()} {precision}"
    return s + (f" ({extra})" if extra else "")


# ----------------------------------------------------------------------------- suites

def gpu_gate(ctx, s):
    """None when QUMO GPU suites may run, else a skip result (unsupported GPU)."""
    if not ctx.gpu:
        return result("skipped", reason=err(E_GPU, "no GPU detected (nvidia-smi / amd-smi / rocm-smi / Apple)"))
    ok, note = gpu_supported(ctx.gpu)
    if not ok and not (s.get("force") or os.environ.get("QUMO_FORCE_GPU")):
        return result("skipped", reason=err(E_GPU, f"{ctx.gpu.get('name')}: {note}; CPU suites and baselines still "
                                                   "run (set QUMO_FORCE_GPU=1 to try anyway)"))
    return None


def suite_gpu_check(ctx, s):
    targets = []
    for t in s.get("targets", ["host"]):
        if t == "host":
            arch = (ctx.gpu or {}).get("arch")
            if arch and gpu_supported(ctx.gpu)[0]:
                targets.append(arch)
        elif t not in targets:
            targets.append(t)
    targets = list(dict.fromkeys(targets))
    if not targets:
        return result("skipped", reason=err(E_GPU, "no host GPU architecture detected and no explicit targets"))
    if not which("make"):
        return result("failed", reason=err(E_TOOL, "make not found"))
    cmd = ["make", f"MOJO={ctx.mojo}", f"GPU_TARGETS={' '.join(targets)}", "gpu-check"]
    ctx.log("gpu_check: targets " + " ".join(targets))
    r = run_cmd(cmd, timeout=s.get("timeout_s", 1800))
    rows = []
    for line in r["stdout"].splitlines():
        m = re.match(r"^(PASS|FAIL)\s+(\S+)\s+(\S+)(?:\s+\((\d+) kernels\))?", line.strip())
        if m:
            rows.append({"kind": "gpu_check", "status": m.group(1), "target": m.group(2), "file": m.group(3),
                         "kernels": int(m.group(4)) if m.group(4) else None})
    n_fail = sum(1 for x in rows if x["status"] == "FAIL")
    status = "timeout" if r["timed_out"] else ("ok" if rows and n_fail == 0 and r["rc"] == 0 else "failed")
    reason = None
    if r["timed_out"]:
        reason = err(E_TIMEOUT, f"make gpu-check exceeded {s.get('timeout_s', 1800)} s")
    elif status == "failed":
        reason = err(E_FAIL, f"{n_fail} gpu-check failure(s), rc={r['rc']}")
    return result(status, rows, reason, commands=[r["cmd"].replace(ROOT + os.sep, "")], targets=targets,
                  summary={"pass": len(rows) - n_fail, "fail": n_fail}, output_tail=tail(r["stdout"] + r["stderr"], 30))


def suite_sv_cpu(ctx, s):
    exe = os.path.join(ROOT, "build", "bench_sv")
    ok, binfo = mojo_build(ctx, exe, "bench/bench_sv.mojo", timeout=s.get("build_timeout_s", 1800))
    if not ok:
        return result("skipped" if binfo.get("skip") else "failed", reason=binfo.get("error"), build=binfo)
    grid, clamp = fit_qubits(qubit_grid(*s.get("qubits", [16, 24])), [("host RAM", ctx.host_budget(), 16 * 1.25)])
    if not grid:
        return result("skipped", reason=err(E_MEMORY, "no width fits the host memory budget"), clamp=clamp)
    reps = int(s.get("reps", 3))
    rows, cmds, problems = [], [], []
    for fam in s.get("families", ["layered"]):
        cmd = [exe, str(grid[-1]), fam, str(grid[0]), str(reps)]
        ctx.log(f"sv_cpu: {fam} {grid[0]}..{grid[-1]} q, reps={reps}")
        r = run_cmd(cmd, timeout=s.get("timeout_s", 1800))
        cmds.append(r["cmd"].replace(ROOT + os.sep, ""))
        t = find_table(parse_bench_csv(r["stdout"]), "qubits", "seconds")
        if t is None:
            problems.append(err(E_TIMEOUT if r["timed_out"] else E_PARSE,
                                f"{fam}: no result table (rc={r['rc']}, timed_out={r['timed_out']})"))
        for row in (t or {"rows": []})["rows"]:
            rows.append({"kind": "statevector", "engine": "qumo", "device": "cpu", "precision": "fp64", "family": fam,
                         "qubits": row.get("qubits"), "gates": row.get("gates"), "depth": row.get("depth"),
                         "seconds": row.get("seconds"), "gates_per_sec": row.get("gates_per_sec"), "reps": reps,
                         "series": label_for("qumo", "cpu", "fp64")})
        if r["timed_out"]:
            problems.append(err(E_TIMEOUT, f"{fam} exceeded {s.get('timeout_s', 1800)} s (kept {len(rows)} rows)"))
    status = "ok" if rows and not problems else ("partial" if rows else "failed")
    return result(status, rows, "; ".join(problems) or None, commands=cmds, build=binfo, clamp=clamp)


def suite_sv_gpu(ctx, s):
    gate = gpu_gate(ctx, s)
    if gate:
        return gate
    if ctx.vendor == "apple" and not ctx.gpu.get("metal_toolchain", True):
        return result("skipped", reason=err(E_GPU, "Metal toolchain missing: run `xcodebuild -downloadComponent MetalToolchain`"))
    lo, hi = s.get("qubits", [16, 26])
    reps = int(s.get("reps", 3))
    rows, frames, cmds, builds, notes, skipped, failures = [], [], [], [], [], [], []
    for prec in s.get("precisions", ["fp32", "fp64"]):
        if prec == "fp64" and ctx.vendor == "apple":
            skipped.append({"precision": prec, "reason": err(E_GPU, "Apple GPUs have no fp64 (Metal; QM-E472)")})
            continue
        bpa = 8 if prec == "fp32" else 16
        plo, phi = s.get(f"qubits_{prec}", [lo, hi])
        grid = [n for n in qubit_grid(16, phi) if n >= plo]  # bench_gpu always sweeps 16, 18, ... nmax
        grid, clamp = fit_qubits(grid, [("GPU memory", ctx.gpu_budget(), bpa * 1.1),
                                        ("host RAM", ctx.host_budget(), 32 * 1.15)])
        if clamp:
            notes.append({"precision": prec, "clamp": clamp})
        if not grid:
            skipped.append({"precision": prec, "reason": err(E_MEMORY, "no width >= 16 fits the memory budget")})
            continue
        exe = os.path.join(ROOT, "build", "gpu", f"bench_gpu_{prec}")
        defines = ["QUMO_GPU"] + (["QUMO_GPU_FP32"] if prec == "fp32" else [])
        ok, binfo = mojo_build(ctx, exe, "bench/bench_gpu.mojo", defines, timeout=s.get("build_timeout_s", 2400))
        builds.append(binfo)
        if not ok:
            (skipped if binfo.get("skip") else failures).append(
                {"precision": prec, "reason": binfo.get("error"), "output_tail": binfo.get("output_tail")})
            continue
        ctx.log(f"sv_gpu: {prec} 16..{grid[-1]} q, reps={reps}")
        r = run_cmd([exe, str(grid[-1]), str(reps)], timeout=s.get("timeout_s", 3600))
        cmds.append(r["cmd"].replace(ROOT + os.sep, ""))
        parsed = parse_bench_csv(r["stdout"])
        info = next((c for c in parsed["comments"] if c.lower().startswith("gpu")), None)
        sv = find_table(parsed, "qubits", "gpu_s")
        fr = find_table(parsed, "code", "gpu_s")
        if sv is None or not sv["rows"] or r["rc"] != 0:
            why = next((c for c in parsed["comments"] if "no usable gpu" in c.lower() or "not compiled" in c.lower()), None)
            exc = first_error_line(r["stdout"] + "\n" + r["stderr"])
            code = E_TIMEOUT if r["timed_out"] else (E_GPU if why else (E_WORKER if exc else E_PARSE))
            entry = {"precision": prec, "output_tail": tail(r["stdout"] + r["stderr"], 20),
                     "reason": err(code, why or exc or f"no GPU rows in output (rc={r['rc']})")}
            (skipped if why else failures).append(entry)
            if sv is None or not sv["rows"]:
                continue
        for row in sv["rows"]:
            n = row.get("qubits")
            if not isinstance(n, int) or n < grid[0] or n > grid[-1]:
                continue
            method = str(row.get("method", ""))
            got = method.rsplit(":", 1)[-1] if ":" in method else prec
            rows.append({"kind": "statevector", "engine": "qumo", "device": "gpu", "precision": got, "family": "layered",
                         "qubits": n, "gates": row.get("gates"), "seconds": row.get("gpu_s"),
                         "gates_per_sec": row.get("gpu_gates_per_s"), "cpu_seconds": row.get("cpu_s"),
                         "speedup_vs_qumo_cpu": row.get("speedup"), "expect_seconds": row.get("gpu_expect_s"),
                         "sample10k_seconds": row.get("gpu_sample10k_s"), "abs_err_expect": row.get("abs_err_expect"),
                         "fidelity": row.get("fidelity"), "method": method, "reps": reps,
                         "emulated": "emulator" in method, "series": label_for("qumo", "gpu", got)})
        for row in (fr or {"rows": []})["rows"]:
            frames.append({"kind": "qec_frames", "engine": "qumo", "device": "gpu", "precision": prec, **row,
                           "series": "QUMO Pauli frames GPU vs CPU"})
        if info:
            notes.append({"precision": prec, "device_info": info})
        if r["timed_out"]:
            notes.append({"precision": prec, "problem": err(E_TIMEOUT, f"exceeded {s.get('timeout_s', 3600)} s")})
    bad_fid = [r_ for r_ in rows if isinstance(r_.get("fidelity"), float) and r_["fidelity"] < 1 - 1e-4]
    if bad_fid:
        notes.append({"problem": err(E_FAIL, f"GPU/CPU fidelity below 1-1e-4 at {[r_['qubits'] for r_ in bad_fid]}")})
    if rows and not skipped and not failures:
        status = "ok"
    else:
        status = "partial" if rows else ("failed" if failures else "skipped")
    reason = "; ".join(dict.fromkeys(x["reason"] for x in failures + skipped)) or None
    return result(status, rows + frames, reason, commands=cmds, builds=builds, notes=notes, skipped=skipped,
                  failures=failures)


def run_worker(ctx, name, python, params, timeout):
    """Run `python run_benchmarks.py --worker name params` -> (messages, run info)."""
    if not os.path.exists(python) and not which(python):
        return [], {"error": err(E_BASELINE, f"interpreter {python} not found")}
    cmd = [python, os.path.abspath(__file__), "--worker", name, json.dumps(params)]
    r = run_cmd(cmd, timeout=timeout)
    msgs = []
    for line in r["stdout"].splitlines():
        if line.startswith(WORKER_TAG):
            try:
                msgs.append(json.loads(line[len(WORKER_TAG):]))
            except ValueError:
                pass
    info = {"command": f"{python} bench/run_benchmarks.py --worker {name} '{json.dumps(params)}'",
            "rc": r["rc"], "seconds": r["seconds"], "timed_out": r["timed_out"]}
    if r["timed_out"]:
        info["error"] = err(E_TIMEOUT, f"worker {name} exceeded {timeout} s")
    elif r["rc"] != 0 and not any(m.get("type") == "done" for m in msgs):
        info["error"] = err(E_WORKER, f"worker {name} exited with {r['rc']}")
        info["output_tail"] = tail(r["stderr"] or r["stdout"], 15)
    return msgs, info


def collect_worker(suite, msgs, info, extra=None):
    rows = [dict(m["row"], **(extra or {})) for m in msgs if m.get("type") == "row"]
    skips = [m for m in msgs if m.get("type") == "skip"]
    meta = {k: v for m in msgs if m.get("type") == "meta" for k, v in m.items() if k != "type"}
    fails = [m.get("error") for m in msgs if m.get("type") == "error"]
    if info.get("error"):
        fails.append(info["error"])
    return rows, skips, meta, fails


def _baseline_suite(ctx, s, worker, combos):
    """Shared driver: combos = [(device, precision, python, params)]."""
    rows, skipped, metas, cmds, problems, clamps = [], [], [], [], [], []
    for device, prec, python, params in combos:
        msgs, info = run_worker(ctx, worker, python, params, s.get("timeout_s", 3600))
        cmds.append(info.get("command"))
        r, sk, meta, fails = collect_worker(worker, msgs, info)
        rows += r
        skipped += [{"device": device, "precision": prec, "reason": x.get("reason")} for x in sk]
        if meta:
            metas.append(dict(meta, device=device, precision=prec))
        if fails and not r:
            skipped.append({"device": device, "precision": prec, "reason": "; ".join(fails)})
        elif fails:
            problems += fails
        if params.get("clamp"):
            clamps.append(params["clamp"])
    status = "ok" if rows and not skipped and not problems else ("partial" if rows else "skipped")
    reason = "; ".join(dict.fromkeys([x["reason"] for x in skipped if x.get("reason")] + problems)) or None
    return result(status, rows, reason, commands=cmds, versions=metas, skipped=skipped, clamps=clamps)


def _grid_params(ctx, s, key, device, prec, reps_default=3):
    rng = s.get(f"{key}_{prec}") or s.get(key) or [16, 24]
    bpa = 8 if prec == "fp32" else 16
    host_need = 2 * 16 * 1.15  # Aer/QUMO statevector in host RAM + result copy (worst case)
    limits = [("host RAM", ctx.host_budget(), host_need if device == "cpu" else bpa * 1.2)]
    if device == "gpu":
        limits.append(("GPU memory", ctx.gpu_budget(), bpa * 1.1))
    grid, clamp = fit_qubits(qubit_grid(*rng), limits)
    return {"qubits": grid, "reps": int(s.get("reps", reps_default)), "depth": 10, "precision": prec,
            "device": device, "clamp": clamp, "verify_qubits": int(s.get("verify_qubits", 8)),
            "families": list(s.get(f"families_{device}") or ["layered"])}


def suite_baseline_aer(ctx, s):
    combos = []
    for device in s.get("devices", ["cpu"]):
        for prec in s.get(f"precisions_{device}", s.get("precisions", ["fp64"])):
            py = ctx.python_for(s, "gpu_python" if device == "gpu" else "python",
                                "QUMO_AER_GPU_PYTHON" if device == "gpu" else None)
            if device == "gpu" and ctx.vendor not in ("nvidia",):
                continue  # qiskit-aer-gpu wheels are CUDA-only; recorded below
            p = _grid_params(ctx, s, "qubits_gpu" if device == "gpu" else "qubits", device, prec)
            if p["qubits"]:
                combos.append((device, prec, py, p))
    res = _baseline_suite(ctx, s, "aer", combos) if combos else result(
        "skipped", reason=err(E_MEMORY, "no Aer width fits the memory budget"))
    if "gpu" in s.get("devices", []) and ctx.vendor != "nvidia":
        # not a failure: the device does not exist on this hardware
        res["not_applicable"] = [{"device": "gpu", "reason": f"qiskit-aer-gpu wheels are CUDA-only (host GPU: {VENDOR_NAMES.get(ctx.vendor, ctx.vendor)})"}]
    return res


def suite_baseline_cudaq(ctx, s):
    combos = []
    py = ctx.python_for(s, "python", "QUMO_CUDAQ_PYTHON")
    for prec in s.get("precisions", ["fp32", "fp64"]):
        if ctx.vendor != "nvidia":
            break
        p = _grid_params(ctx, s, "qubits", "gpu", prec)
        p["target"] = "nvidia"
        if p["qubits"]:
            combos.append(("gpu", prec, py, p))
    if s.get("cpu", False):
        p = _grid_params(ctx, s, "qubits_cpu", "cpu", "fp64")
        p["target"] = "qpp-cpu"
        if p["qubits"]:
            combos.append(("cpu", "fp64", py, p))
    if not combos:
        why = f"CUDA-Q GPU targets need an NVIDIA GPU (host GPU: {VENDOR_NAMES.get(ctx.vendor, ctx.vendor)})"
        if ctx.vendor == "nvidia":  # NVIDIA present but nothing fits: a real problem
            return result("skipped", reason=err(E_MEMORY, "no CUDA-Q width fits the memory budget"))
        return result("skipped", reason="not applicable: " + why, not_applicable=[{"device": "gpu", "reason": why}])
    return _baseline_suite(ctx, s, "cudaq", combos)


def suite_baseline_lightning(ctx, s):
    combos = []
    py = ctx.python_for(s, "python", "QUMO_LIGHTNING_PYTHON")
    vendor_dev = {"lightning.gpu": "nvidia", "lightning.amdgpu": "amd"}
    n_a = []
    for dev in s.get("devices", ["lightning.qubit"]):
        need = vendor_dev.get(dev)
        if need and ctx.vendor != need:
            n_a.append({"device": dev, "reason": f"{dev} needs an {VENDOR_NAMES[need]} GPU "
                                                 f"(host GPU: {VENDOR_NAMES.get(ctx.vendor, ctx.vendor)})"})
            continue
        kind = "cpu" if dev == "lightning.qubit" else "gpu"
        for prec in s.get(f"precisions_{kind}", s.get("precisions", ["fp64"])):
            p = _grid_params(ctx, s, "qubits_gpu" if kind == "gpu" else "qubits", kind, prec)
            p["pl_device"] = dev
            if p["qubits"]:
                combos.append((kind, prec, py, p))
    res = _baseline_suite(ctx, s, "lightning", combos) if combos else result(
        "skipped", reason=err(E_BASELINE, "; ".join(x["reason"] for x in n_a) or "no Lightning device selected"))
    res["not_applicable"] = n_a
    return res


def suite_neutral(ctx, s):
    py = ctx.python_for(s, "python")
    so = os.path.join(ROOT, "build", "qumo_py.so")
    ok, binfo = mojo_build(ctx, so, "python/qumo_py.mojo", extra=["--emit", "shared-lib"],
                           timeout=s.get("build_timeout_s", 1800))
    if not ok:
        return result("skipped" if binfo.get("skip") else "failed", reason=binfo.get("error"), build=binfo)
    out = os.path.splitext(ctx.out_json)[0] + "_neutral.jsonl"
    if os.path.exists(out):
        os.remove(out)
    cmd = [py, os.path.join("bench", "neutral", "run_neutral.py"), "--circuits", os.path.join("bench", "neutral", "circuits"),
           "--sdks", ",".join(s.get("sdks", ["qumo", "qiskit", "tket"])), "--targets", ",".join(s.get("targets", ["sc_grid"])),
           "--levels", ",".join(s.get("levels", ["default"])), "--max-qubits", str(s.get("max_qubits", 6)),
           "--filter", s.get("filter", "qb_small__"), "--timeout", str(s.get("job_timeout_s", 120)), "--out", out]
    ctx.log("neutral: " + " ".join(cmd[2:]))
    r = run_cmd(cmd, timeout=s.get("timeout_s", 1800))
    recs = []
    if os.path.exists(out):
        with open(out) as fh:
            recs = [json.loads(x) for x in fh if x.strip()]
    rows = []
    for sdk in sorted({x["sdk"] for x in recs}):
        mine = [x for x in recs if x["sdk"] == sdk]
        okr = [x for x in mine if x.get("status") == "ok"]
        rows.append({"kind": "compiler", "engine": sdk, "jobs": len(mine),
                     **{f"n_{st}": sum(1 for x in mine if x.get("status") == st) for st in
                        ("ok", "unsupported", "invalid", "incorrect", "error", "timeout")},
                     "median_compile_s": median([x["wall_s"] for x in okr if "wall_s" in x]),
                     "median_n2q": median([x["n2q"] for x in okr if isinstance(x.get("n2q"), (int, float))]),
                     "mean_esp": (statistics.fmean([x["esp"] for x in okr if isinstance(x.get("esp"), (int, float))])
                                  if any(isinstance(x.get("esp"), (int, float)) for x in okr) else None)})
    status = "timeout" if r["timed_out"] else ("ok" if rows and r["rc"] == 0 else ("partial" if rows else "failed"))
    reason = None
    if status != "ok":
        reason = err(E_TIMEOUT if r["timed_out"] else E_FAIL, f"run_neutral.py rc={r['rc']}")
    return result(status, rows, reason, commands=[r["cmd"].replace(ROOT + os.sep, "")], records_file=os.path.basename(out),
                  output_tail=tail(r["stderr"] if r["rc"] else r["stdout"], 15))


def suite_qec(ctx, s):
    py = ctx.python_for(s, "python")
    cmd = [py, os.path.join("bench", "bench_qec.py"), *map(str, s.get("args", ["--part", "A", "--shots", "20000"]))]
    env = dict(os.environ)
    env.setdefault("PYTHONUNBUFFERED", "1")
    ctx.log("qec: " + " ".join(cmd[1:]))
    r = run_cmd(cmd, timeout=s.get("timeout_s", 1800), env=env)
    rows = []
    for i, t in enumerate(parse_markdown_tables(r["stdout"])):
        for row in t["rows"]:
            rows.append({"kind": "qec", "table": i, **row})
    status = "timeout" if r["timed_out"] else ("ok" if rows and r["rc"] == 0 else ("partial" if rows else "failed"))
    reason = None
    if status != "ok":
        reason = err(E_TIMEOUT if r["timed_out"] else E_FAIL, f"bench_qec.py rc={r['rc']}")
    return result(status, rows, reason, commands=[r["cmd"].replace(ROOT + os.sep, "")],
                  output_tail=tail(r["stdout"] + ("\n" + r["stderr"] if r["rc"] else ""), 25))


def suite_tests(ctx, s):
    rows, cmds, problems = [], [], []
    for target in s.get("targets", ["test"]):
        gpu = target == "test-gpu"
        if gpu and not ctx.gpu:
            problems.append(err(E_GPU, "test-gpu skipped: no GPU detected"))
            continue
        files = s.get("files_gpu" if gpu else "files")
        if files:  # explicit subset: same PASS rule as the Makefile (last line contains OK)
            for f in files:
                cmd = [ctx.mojo, "run", "-I", ".", *(["-D", "QUMO_GPU"] if gpu else []), f]
                r = run_cmd(cmd, timeout=s.get("file_timeout_s", 900))
                cmds.append(r["cmd"].replace(ROOT + os.sep, ""))
                last = tail(r["stdout"] + r["stderr"], 1)
                rows.append({"kind": "test", "target": target, "file": f,
                             "status": "PASS" if "OK" in last and not r["timed_out"] else "FAIL",
                             "detail": None if "OK" in last else ("timeout" if r["timed_out"] else last[-200:]),
                             "seconds": r["seconds"]})
        else:
            cmd = ["make", f"MOJO={ctx.mojo}", target]
            ctx.log(f"tests: make {target}")
            r = run_cmd(cmd, timeout=s.get("timeout_s", 3600))
            cmds.append(r["cmd"].replace(ROOT + os.sep, ""))
            for line in r["stdout"].splitlines():
                m = re.match(r"^(PASS|FAIL)\s+(\S+?):?(?:\s+(.*))?$", line.strip())
                if m:
                    rows.append({"kind": "test", "target": target, "file": m.group(2), "status": m.group(1),
                                 "detail": (m.group(3) or None)})
            if r["timed_out"]:
                problems.append(err(E_TIMEOUT, f"make {target} exceeded {s.get('timeout_s', 3600)} s"))
    summary = {}
    for t in {x["target"] for x in rows}:
        mine = [x for x in rows if x["target"] == t]
        summary[t] = {"pass": sum(1 for x in mine if x["status"] == "PASS"),
                      "fail": sum(1 for x in mine if x["status"] == "FAIL")}
    n_fail = sum(v["fail"] for v in summary.values())
    if n_fail:
        problems.append(err(E_FAIL, f"{n_fail} test file(s) failed"))
    status = "ok" if rows and not problems else ("partial" if rows else "failed")
    return result(status, rows, "; ".join(problems) or None, commands=cmds, summary=summary)


SUITES = {  # execution order
    "gpu_check": (suite_gpu_check, "make gpu-check: cross-compile every GPU kernel for the host arch (+ targets)"),
    "sv_cpu": (suite_sv_cpu, "QUMO CPU state vector (build/bench_sv): layered / qft / random families"),
    "sv_gpu": (suite_sv_gpu, "QUMO GPU state vector fp32/fp64 + QEC Pauli frames (build/gpu/bench_gpu_*)"),
    "baseline_aer": (suite_baseline_aer, "Qiskit Aer statevector, CPU and GPU (qiskit-aer-gpu, NVIDIA only)"),
    "baseline_cudaq": (suite_baseline_cudaq, "NVIDIA CUDA-Q target nvidia fp32/fp64 (+ optional qpp-cpu)"),
    "baseline_lightning": (suite_baseline_lightning, "PennyLane Lightning: qubit (CPU), gpu (NVIDIA), amdgpu (AMD)"),
    "neutral": (suite_neutral, "hardware-neutral compiler benchmark subset (bench/neutral)"),
    "qec": (suite_qec, "QEC cross-validation subset (bench/bench_qec.py vs Stim + PyMatching)"),
    "tests": (suite_tests, "make test / make test-gpu (or listed files): pass/fail counts"),
}


# ----------------------------------------------------------------------------- baseline workers
# Run inside the baseline's own interpreter: only stdlib at import time, the
# SDK is imported inside the worker function.

def emit(**kw):
    print(WORKER_TAG + json.dumps(kw, default=str), flush=True)


def _dist_versions(*dists):
    from importlib import metadata
    out = {}
    for d in dists:
        try:
            out[d] = metadata.version(d)
        except metadata.PackageNotFoundError:
            pass
    return out


def layered_angles(n, depth):
    ry = [0.1 + 0.01 * (q + d) for d in range(depth) for q in range(n)]
    rz = [0.2 + 0.03 * (q * d) for d in range(depth) for q in range(n)]
    return ry, rz


def layered_ops(n, depth):
    """The exact gate sequence of bench/bench_sv.mojo::layered (shared by all baselines)."""
    ops = []
    for d in range(depth):
        for q in range(n):
            ops.append(("ry", 0.1 + 0.01 * (q + d), q))
            ops.append(("rz", 0.2 + 0.03 * (q * d), q))
        for q in range(d % 2, n - 1, 2):
            ops.append(("cx", q, q + 1))
    return ops


def reference_state(n, depth):
    """Tiny numpy reference (little-endian: bit q of the index = qubit q)."""
    import numpy as np
    psi = np.zeros([2] * n, dtype=complex)
    psi[(0,) * n] = 1.0
    ax = lambda q: n - 1 - q  # noqa: E731  axis of qubit q in the C-ordered tensor
    for op in layered_ops(n, depth):
        if op[0] in ("ry", "rz"):
            t, q = op[1], op[2]
            c, s_ = math.cos(t / 2), math.sin(t / 2)
            u = (np.array([[c, -s_], [s_, c]]) if op[0] == "ry"
                 else np.array([[np.exp(-0.5j * t), 0], [0, np.exp(0.5j * t)]]))
            psi = np.moveaxis(np.tensordot(u, psi, axes=([1], [ax(q)])), 0, ax(q))
        else:
            c_, t_ = op[1], op[2]
            idx = [slice(None)] * n
            idx[ax(c_)] = 1
            sub = psi[tuple(idx)]
            t_ax = ax(t_) - (1 if ax(t_) > ax(c_) else 0)
            psi[tuple(idx)] = np.flip(sub, axis=t_ax)
    return psi.reshape(-1)


def state_fidelity(a, b):
    """max over the two qubit orders (engines differ in endianness)."""
    import numpy as np
    a = np.asarray(a, dtype=complex).reshape(-1)
    b = np.asarray(b, dtype=complex).reshape(-1)
    n = int(round(math.log2(len(a))))
    rev = np.asarray(b.reshape([2] * n).transpose(list(range(n))[::-1])).reshape(-1)
    f1 = abs(np.vdot(a, b)) ** 2
    f2 = abs(np.vdot(a, rev)) ** 2
    return (float(f1), "little-endian") if f1 >= f2 else (float(f2), "big-endian")


def _aer_family_builder(family):
    """QuantumCircuit builder for a circuit family. `layered` comes from layered_ops
    (shared with every baseline); `qft` / `random` are imported from
    bench/bench_sv_aer.py, the gate-for-gate Qiskit twin of bench/bench_sv.mojo."""
    from qiskit import QuantumCircuit
    if family == "layered":
        def build(n, depth):
            c = QuantumCircuit(n)
            for op in layered_ops(n, depth):
                if op[0] == "ry":
                    c.ry(op[1], op[2])
                elif op[0] == "rz":
                    c.rz(op[1], op[2])
                else:
                    c.cx(op[1], op[2])
            return c
        return build
    import importlib.util
    spec = importlib.util.spec_from_file_location("qumo_bench_sv_aer", os.path.join(ROOT, "bench", "bench_sv_aer.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return lambda n, depth: mod.make(family, n)


def worker_aer(p):
    import qiskit
    import qiskit_aer
    from qiskit_aer import AerSimulator

    devices = list(AerSimulator().available_devices())
    emit(type="meta", engine="aer", versions=dict(_dist_versions("qiskit-aer", "qiskit-aer-gpu", "qiskit-aer-gpu-cu11"),
                                                  qiskit=qiskit.__version__, qiskit_aer_module=qiskit_aer.__version__),
         available_devices=devices)
    dev = p["device"].upper()
    if dev not in devices:
        emit(type="skip", reason=err(E_BASELINE, f"Aer device {dev} unavailable (available: {devices}); "
                                     "GPU needs qiskit-aer-gpu on Linux x86_64 + NVIDIA"))
        emit(type="done")
        return
    sim = AerSimulator(method="statevector", device=dev, precision="double" if p["precision"] == "fp64" else "single")
    layered = _aer_family_builder("layered")
    vq = p["verify_qubits"]
    c = layered(vq, p["depth"])
    c.save_statevector()
    sv = sim.run(c).result().get_statevector()
    fid, order = state_fidelity(reference_state(vq, p["depth"]), sv)
    emit(type="meta", verify={"qubits": vq, "fidelity": fid, "order": order})
    c = layered(p["qubits"][0], p["depth"])
    c.save_amplitudes([0])
    sim.run(c).result()  # warm-up (library load, GPU context)
    for family in p.get("families", ["layered"]):
        build = _aer_family_builder(family)
        for n in p["qubits"]:
            c = build(n, p["depth"])
            gates = len(c.data)
            c.save_amplitudes([0])  # forces the full evolution, copies one amplitude to the host
            times = []
            for _ in range(p["reps"]):
                t0 = time.perf_counter()
                res = sim.run(c).result()
                times.append(time.perf_counter() - t0)
                if not res.success:
                    emit(type="error", error=f"Aer run failed at {n} qubits ({family}): {res.status}")
                    emit(type="done")
                    return
            t = median(times)
            emit(type="row", row={"kind": "statevector", "engine": "aer", "device": p["device"],
                                  "precision": p["precision"], "family": family, "qubits": n, "gates": gates,
                                  "seconds": t, "gates_per_sec": gates / t, "reps": p["reps"], "times": times,
                                  "series": label_for("aer", p["device"], p["precision"])})
    emit(type="done")


def worker_cudaq(p):
    import cudaq
    target = p.get("target", "nvidia")
    try:
        if target == "nvidia":
            cudaq.set_target("nvidia", option=p["precision"])
        else:
            cudaq.set_target(target)
    except Exception as e:  # no GPU / no cuStateVec
        emit(type="meta", engine="cudaq", versions=_dist_versions("cudaq", "cuda-quantum-cu12", "cuda-quantum-cu13"))
        emit(type="skip", reason=err(E_BASELINE, f"cudaq.set_target({target!r}, {p['precision']}) failed: {e}"))
        emit(type="done")
        return
    emit(type="meta", engine="cudaq", versions=_dist_versions("cudaq", "cuda-quantum-cu12", "cuda-quantum-cu13"),
         target=target,
         num_gpus=(cudaq.num_available_gpus() if hasattr(cudaq, "num_available_gpus") else None))
    depth = p["depth"]
    try:
        @cudaq.kernel
        def layered_kernel(n: int, depth: int, ry_angles: list[float], rz_angles: list[float]):
            q = cudaq.qvector(n)
            for d in range(depth):
                for i in range(n):
                    ry(ry_angles[d * n + i], q[i])  # noqa: F821  (CUDA-Q kernel builtins)
                    rz(rz_angles[d * n + i], q[i])  # noqa: F821
                for i in range(d % 2, n - 1, 2):
                    x.ctrl(q[i], q[i + 1])  # noqa: F821

        def state(n):
            ry, rz = layered_angles(n, depth)
            return cudaq.get_state(layered_kernel, n, depth, ry, rz)
        style = "kernel"
        state(2)
    except Exception as e:  # fall back to the kernel builder (unrolled per width)
        emit(type="meta", kernel_note=f"@cudaq.kernel failed ({type(e).__name__}: {e}); using cudaq.make_kernel")
        cache = {}

        def state(n):
            if n not in cache:
                k = cudaq.make_kernel()
                q = k.qalloc(n)
                for op in layered_ops(n, depth):
                    if op[0] == "ry":
                        k.ry(op[1], q[op[2]])
                    elif op[0] == "rz":
                        k.rz(op[1], q[op[2]])
                    else:
                        k.cx(q[op[1]], q[op[2]])
                cache.clear()
                cache[n] = k
            return cudaq.get_state(cache[n])
        style = "builder"
    import numpy as np

    def amplitudes(st, n):
        try:
            a = np.array(st, dtype=complex).reshape(-1)
            if a.size == 2 ** n:
                return a
        except Exception:  # device-resident state without a host buffer
            pass
        return np.array([complex(st[i]) for i in range(2 ** n)])
    vq = p["verify_qubits"]
    amps = amplitudes(state(vq), vq)
    fid, order = state_fidelity(reference_state(vq, depth), amps)
    emit(type="meta", verify={"qubits": vq, "fidelity": fid, "order": order}, kernel_style=style)
    gates_of = lambda n: len(layered_ops(n, depth))  # noqa: E731
    for n in p["qubits"]:
        s0 = state(n)  # warm-up per width (JIT / allocation)
        del s0
        times = []
        for _ in range(p["reps"]):
            t0 = time.perf_counter()
            s1 = state(n)
            try:
                _ = s1[0]  # touch one amplitude: forces completion on the device
            except Exception:
                pass
            times.append(time.perf_counter() - t0)
            del s1
        t = median(times)
        dev = "gpu" if target == "nvidia" else "cpu"
        emit(type="row", row={"kind": "statevector", "engine": "cudaq", "device": dev, "precision": p["precision"],
                              "family": "layered", "qubits": n, "gates": gates_of(n), "seconds": t,
                              "gates_per_sec": gates_of(n) / t, "reps": p["reps"], "times": times,
                              "series": label_for("cudaq", dev, p["precision"], target if dev == "cpu" else "")})
    emit(type="done")


def worker_lightning(p):
    import numpy as np
    import pennylane as qml
    dev_name = p["pl_device"]
    versions = {"pennylane": qml.__version__}
    from importlib import metadata
    for dist in ("pennylane-lightning", "pennylane-lightning-gpu", "pennylane-lightning-amdgpu",
                 "pennylane-lightning-kokkos"):
        try:
            versions[dist] = metadata.version(dist)
        except metadata.PackageNotFoundError:
            pass
    emit(type="meta", engine="lightning", versions=versions, pl_device=dev_name)
    dtype = np.complex64 if p["precision"] == "fp32" else np.complex128
    depth = p["depth"]

    def make(n, measure):
        try:
            dev = qml.device(dev_name, wires=n, c_dtype=dtype)
        except Exception as e:
            raise RuntimeError(f"{dev_name}: {e}")

        @qml.qnode(dev)
        def circ():
            for op in layered_ops(n, depth):
                if op[0] == "ry":
                    qml.RY(op[1], wires=op[2])
                elif op[0] == "rz":
                    qml.RZ(op[1], wires=op[2])
                else:
                    qml.CNOT(wires=[op[1], op[2]])
            return qml.state() if measure == "state" else qml.expval(qml.PauliZ(0))
        return circ
    try:
        vq = p["verify_qubits"]
        st = make(vq, "state")()
    except Exception as e:
        emit(type="skip", reason=err(E_BASELINE, f"{dev_name} unavailable: {e}"))
        emit(type="done")
        return
    fid, order = state_fidelity(reference_state(vq, depth), st)
    emit(type="meta", verify={"qubits": vq, "fidelity": fid, "order": order})
    gates_of = lambda n: len(layered_ops(n, depth))  # noqa: E731
    kind = "cpu" if dev_name in ("lightning.qubit",) else "gpu"
    for n in p["qubits"]:
        circ = make(n, "expval")  # <Z0>: device-side reduction, no 2^n host copy
        circ()  # warm-up per width (allocation, tape construction caches)
        times = []
        for _ in range(p["reps"]):
            t0 = time.perf_counter()
            circ()
            times.append(time.perf_counter() - t0)
        t = median(times)
        emit(type="row", row={"kind": "statevector", "engine": "lightning", "device": kind, "precision": p["precision"],
                              "family": "layered", "qubits": n, "gates": gates_of(n), "seconds": t,
                              "gates_per_sec": gates_of(n) / t, "reps": p["reps"], "times": times,
                              "series": label_for("lightning", kind, p["precision"], dev_name)})
    emit(type="done")


WORKERS = {"aer": worker_aer, "cudaq": worker_cudaq, "lightning": worker_lightning}


def worker_main(name, params_json):
    try:
        WORKERS[name](json.loads(params_json))
    except ImportError as e:
        emit(type="skip", reason=err(E_BASELINE, f"{name}: {e} (interpreter {sys.executable})"))
        emit(type="done")
    except Exception as e:
        import traceback
        emit(type="error", error=err(E_WORKER, f"{name}: {type(e).__name__}: {e}"),
             trace=traceback.format_exc()[-1500:])
        emit(type="done")


# ----------------------------------------------------------------------------- output

ROW_FIRST = ["suite", "kind", "series", "engine", "device", "precision", "family", "qubits", "gates", "seconds",
             "gates_per_sec", "reps"]


def scrub(text):
    """Results get shared: never leak local absolute paths / user names."""
    home = os.path.expanduser("~")
    text = text.replace(ROOT + os.sep, "").replace(ROOT, ".")
    return text.replace(home, "~") if home and home != os.sep else text


def write_outputs(doc, out_json):
    tmp = out_json + ".tmp"
    with open(tmp, "w") as fh:
        fh.write(scrub(json.dumps(doc, indent=1, default=str)))
    os.replace(tmp, out_json)
    rows = [dict(r, suite=name) for name, s in doc["suites"].items() for r in s.get("rows", [])]
    keys = list(ROW_FIRST)
    for r in rows:
        for k in r:
            if k not in keys and k != "times":
                keys.append(k)
    with open(os.path.splitext(out_json)[0] + ".csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: scrub(str(r[k])) if isinstance(r.get(k), str) else r.get(k) for k in keys})


def default_out(cfg_name, env):
    g = env["gpus"][0]["name"] if env["gpus"] else "cpu"
    slug = re.sub(r"[^a-z0-9]+", "-", (g or "gpu").lower()).strip("-")[:32]
    stamp = time.strftime("%Y%m%d-%H%M%S")
    return f"qumo_bench_{cfg_name}_{slug}_{stamp}.json"


def print_summary(doc):
    print("\n=== QUMO benchmark summary ===")
    print(f"config: {doc['config']['name']}   gpu: {', '.join(g.get('name') or '?' for g in doc['environment']['gpus']) or 'none'}"
          f"   load avg start/end: {doc['environment'].get('loadavg_start')} / {doc.get('loadavg_end')}")
    for name, s in doc["suites"].items():
        reason = (" - " + s["reason"][:160]) if s.get("reason") else ""
        print(f"  {name:20s} {s['status']:8s} {len(s.get('rows', [])):4d} rows {s.get('duration_s', 0):8.1f} s{reason}")
    series = {}
    for s in doc["suites"].values():
        for r in s.get("rows", []):
            if r.get("kind") == "statevector" and r.get("family", "layered") == "layered" and r.get("seconds"):
                series.setdefault(r["series"], {})[r["qubits"]] = r["seconds"]
    if series:
        qs = sorted({q for v in series.values() for q in v})
        print("\nlayered circuit, median seconds:")
        print(f"  {'series':34s}" + "".join(f"{q:>9d}" for q in qs))
        for name in sorted(series):
            print(f"  {name[:34]:34s}" + "".join(f"{series[name][q]:9.4f}" if q in series[name] else f"{'-':>9s}" for q in qs))


# ----------------------------------------------------------------------------- main

def list_everything(env):
    print("Suites (execution order):")
    for k, (_, d) in SUITES.items():
        print(f"  {k:20s} {d}")
    print("\nConfigs (bench/configs):")
    for p in sorted(glob.glob(os.path.join(CONFIG_DIR, "*.json"))):
        try:
            with open(p) as fh:
                c = json.load(fh)
            on = [k for k, v in c.get("suites", {}).items() if v.get("enabled", True)]
            print(f"  {os.path.basename(p)[:-5]:14s} {c.get('description', '')[:70]}")
            print(f"  {'':14s} suites: {', '.join(on)}")
        except ValueError:
            print(f"  {os.path.basename(p)}: invalid JSON")
    print("\nDetected GPUs:")
    for g in env["gpus"] or [{"name": "none"}]:
        print("  " + ", ".join(f"{k}={v}" for k, v in g.items() if k != "source"))
    print(f"\n--config auto -> {auto_config(env['gpus'])}")


def build_only(ctx, suites):
    """--build-only: compile what the selected suites need (same rules and gates as a run)."""
    ok_all = True

    def report(ok, info):
        nonlocal ok_all
        print(("OK   " if ok else "FAIL ") + json.dumps(info)[:300], flush=True)
        ok_all &= ok
    scfg = ctx.cfg["suites"]
    if "sv_cpu" in suites:
        report(*mojo_build(ctx, os.path.join(ROOT, "build", "bench_sv"), "bench/bench_sv.mojo",
                           timeout=scfg.get("sv_cpu", {}).get("build_timeout_s", 1800)))
    if "sv_gpu" in suites:
        gate = gpu_gate(ctx, scfg.get("sv_gpu", {}))
        if gate:
            print("SKIP sv_gpu build: " + gate["reason"])
        else:
            for prec in scfg.get("sv_gpu", {}).get("precisions", ["fp32"]):
                if prec == "fp64" and ctx.vendor == "apple":
                    print("SKIP bench_gpu_fp64: Apple GPUs have no fp64 (QM-E472)")
                    continue
                report(*mojo_build(ctx, os.path.join(ROOT, "build", "gpu", f"bench_gpu_{prec}"), "bench/bench_gpu.mojo",
                                   ["QUMO_GPU"] + (["QUMO_GPU_FP32"] if prec == "fp32" else []),
                                   timeout=scfg.get("sv_gpu", {}).get("build_timeout_s", 2400)))
    if "neutral" in suites:
        report(*mojo_build(ctx, os.path.join(ROOT, "build", "qumo_py.so"), "python/qumo_py.mojo",
                           extra=["--emit", "shared-lib"], timeout=scfg.get("neutral", {}).get("build_timeout_s", 1800)))
    return 0 if ok_all else 1


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("--config", default="auto", help="config path, preset name (bench/configs/<name>.json) or 'auto'")
    ap.add_argument("--out", default=os.path.join("bench", "results", "cloud"),
                    help="result .json path, or a directory (auto-named file)")
    ap.add_argument("--suites", default="", help="comma list; overrides the config's enabled suites")
    ap.add_argument("--quick", action="store_true", help="small sizes, 1 repetition (smoke run)")
    ap.add_argument("--list", action="store_true", help="list suites, configs, detected GPUs and exit")
    ap.add_argument("--select-config", metavar="GPU_NAME",
                    help="print the preset `--config auto` would pick for this GPU name and exit "
                         "(refine with --gpu-memory-gib / --gpu-vendor / --gpu-arch)")
    ap.add_argument("--gpu-memory-gib", type=float, default=None, help=argparse.SUPPRESS)
    ap.add_argument("--gpu-vendor", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--gpu-arch", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--build-only", action="store_true", help="build the binaries the selected suites need, then exit")
    ap.add_argument("--no-build", action="store_true", help="never build; suites with missing binaries are skipped")
    ap.add_argument("--mojo", default=os.environ.get("QUMO_MOJO", os.path.join(ROOT, ".venv", "bin", "mojo")))
    ap.add_argument("--label", default="", help="free-text label stored in the result (e.g. 'colab pro, high-RAM')")
    ap.add_argument("--worker", help=argparse.SUPPRESS)
    ap.add_argument("worker_params", nargs="?", help=argparse.SUPPRESS)
    a = ap.parse_args(argv)

    if a.worker:
        worker_main(a.worker, a.worker_params or "{}")
        return 0
    if a.select_config == "auto":  # detect the local GPU(s): line 1 = preset, then one '#' line per GPU
        gpus = detect_gpus()
        print(auto_config(gpus))
        for g in gpus:
            ok, note = gpu_supported(g)
            mem = f"{g['memory_bytes'] / 2 ** 30:.1f} GiB" if g.get("memory_bytes") else "memory ?"
            print(f"# {g.get('vendor')} GPU {g.get('index')}: {g.get('name') or '?'} ({g.get('arch') or 'arch ?'}, {mem}"
                  + (f", driver {g['driver']}" if g.get("driver") else "") + (f", ROCm {g['rocm']}" if g.get("rocm") else "")
                  + f"); QUMO GPU kernels: {'supported' if ok else 'NOT supported'}" + (f" - {note}" if note else ""))
        if not gpus:
            print("# no GPU detected (nvidia-smi / amd-smi / rocm-smi / rocminfo / Apple silicon)")
        return 0
    if a.select_config is not None:
        print(select_config(a.select_config, a.gpu_vendor, a.gpu_memory_gib, a.gpu_arch))
        return 0
    os.chdir(ROOT)
    env = collect_env(ROOT, a.mojo)
    if a.list:
        list_everything(env)
        return 0
    try:
        cfg_path = resolve_config_path(a.config, env["gpus"])
        cfg = load_config(cfg_path)
        cfg_name = cfg.get("name") or os.path.basename(cfg_path)[:-5]
        enabled = [k for k in SUITES if k in cfg["suites"] and cfg["suites"][k].get("enabled", True)]
        if a.suites:
            req = [x.strip() for x in a.suites.split(",") if x.strip()]
            bad = [x for x in req if x not in SUITES]
            if bad:
                raise BenchError(err(E_CONFIG, f"unknown suite(s) {bad}; known: {list(SUITES)}"))
            enabled = [k for k in SUITES if k in req]
    except BenchError as e:
        print(str(e), file=sys.stderr)
        return 2
    out = a.out
    if out.endswith(os.sep) or os.path.isdir(out) or not out.endswith(".json"):
        out = os.path.join(out, default_out(cfg_name + ("_quick" if a.quick else ""), env))
    out = os.path.abspath(out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    ctx = Ctx(a, cfg, env, out)
    if a.build_only:
        return build_only(ctx, enabled)

    doc = {"schema": SCHEMA, "label": a.label, "config": {"name": cfg_name, "path": os.path.relpath(cfg_path, ROOT),
                                                          "quick": a.quick, "content": cfg},
           "environment": env, "suites": {}, "started_utc": now_iso()}
    ctx.log(f"config {cfg_name} ({os.path.relpath(cfg_path, ROOT)}); suites: {', '.join(enabled) or 'none'}")
    ctx.log(f"GPU: {', '.join((g.get('name') or '?') + ' [' + g['vendor'] + ']' for g in env['gpus']) or 'none'}; "
            f"load avg {env['loadavg_start']}; output {out}")
    for name in enabled:
        scfg = cfg["suites"].get(name, {})
        if a.quick:
            scfg = apply_quick(scfg)
        t0 = time.perf_counter()
        la0 = loadavg()
        ctx.log(f"--- suite {name}")
        try:
            res = SUITES[name][0](ctx, scfg)
        except BenchError as e:
            res = result("failed", reason=str(e))
        except Exception as e:  # a suite bug must never abort the whole run
            import traceback
            res = result("failed", reason=err(E_WORKER, f"{name}: {type(e).__name__}: {e}"),
                         trace=traceback.format_exc()[-2000:])
        res["duration_s"] = round(time.perf_counter() - t0, 2)
        res["loadavg_before"], res["loadavg_after"] = la0, loadavg()
        res["config"] = scfg
        doc["suites"][name] = res
        ctx.log(f"    {name}: {res['status']} ({len(res.get('rows', []))} rows, {res['duration_s']} s)"
                + (f" - {res['reason'][:200]}" if res.get("reason") else ""))
        doc["finished_utc"], doc["loadavg_end"] = now_iso(), loadavg()
        write_outputs(doc, out)
    doc["finished_utc"], doc["loadavg_end"] = now_iso(), loadavg()
    write_outputs(doc, out)
    print_summary(doc)
    print(f"\nresults: {out}\n         {os.path.splitext(out)[0] + '.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
