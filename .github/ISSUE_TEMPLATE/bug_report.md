---
name: Bug report
about: Something in QUMO gives a wrong result, crashes, or does not build
title: ""
labels: bug
---

**What happened**

<!-- One or two sentences. Include the error code if there is one (QM-Exxx). -->

**How to reproduce**

<!-- The smallest OpenQASM file or Mojo / Python script that shows the problem, and the exact command you ran. -->

```
build/qumo run my.qasm --shots 1000 --seed 7
```

**What you expected**

**Full output**

```
(paste here; the first line "Failed to initialize Crashpad..." is harmless and can be left out)
```

**Environment**

- QUMO commit (`git rev-parse --short HEAD`):
- Mojo (`.venv/bin/mojo --version`):
- OS and CPU (for example macOS 15.5, Apple M3; Ubuntu 24.04, x86-64):
- Interface: CLI / Mojo / Python (`qumo_py`) / MCP / hardware plugin
- GPU build? (`-D QUMO_GPU`, which GPU):
