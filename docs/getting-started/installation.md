# Installation Guide

## Requirements
- Python 3.10, 3.11, or 3.12
- PyTorch >= 2.0.0
- Transformers >= 4.30.0

## Installing via pip

```bash
pip install continuous-legal-memory
```

### Optional Extras

- **Telemetry & OpenTelemetry Support**:
  ```bash
  pip install "continuous-legal-memory[telemetry]"
  ```
- **Development & Testing Suite**:
  ```bash
  pip install "continuous-legal-memory[dev]"
  ```

---

## Verifying Installation

Run a quick inline Python check:

```python
import continuous_legal_memory
print(continuous_legal_memory.__version__)
```
