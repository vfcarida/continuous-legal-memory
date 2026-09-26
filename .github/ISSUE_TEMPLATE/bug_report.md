---
name: Bug Report
about: Create a report to help us improve Continuous Legal Memory
title: "[BUG]: "
labels: ["bug"]
assignees: ""
---

### Describe the Bug
A clear and concise description of what the bug is.

### Reproduction Steps
Steps to reproduce the behavior:
1. Initialize `LegalMemoryOrchestrator(...)` with ...
2. Ingest rule via `orchestrator.update_memory(...)`
3. Call `orchestrator.predict(...)`
4. See error or unexpected output

### Minimal Reproducible Example
```python
from continuous_legal_memory.orchestrator import LegalMemoryOrchestrator

orchestrator = LegalMemoryOrchestrator(value_dim=2, engine_mode="structured")
# Ingest and query
```

### Expected Behavior
A clear and concise description of what you expected to happen.

### Environment Information
- **OS**: [e.g. Ubuntu 22.04, macOS Sonoma, Windows 11]
- **Python Version**: [e.g. 3.10.12, 3.11.5, 3.12.0]
- **Continuous Legal Memory Version**: [e.g. 0.1.0]
- **PyTorch Version**: [e.g. 2.2.0]
- **Hardware**: [e.g. CPU, Apple Silicon M2, NVIDIA RTX 4090]

### Additional Context
Add any other context, logs, or error tracebacks here.
