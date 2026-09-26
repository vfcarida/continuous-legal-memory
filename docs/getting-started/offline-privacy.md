# Offline Edge Deployment & Attorney-Client Privilege

Continuous Legal Memory is engineered specifically for regulated law firms, corporate legal departments, and government institutions where confidential documents cannot leave the local network.

---

## Strict Loopback Enforcement

The `OllamaAdapter` includes strict privacy enforcement to prevent accidental data leakage:

- **Loopback Enforcement**: Connection URLs are strictly restricted to `127.0.0.1` and `localhost`.
- **Exfiltration Blocking**: Any attempt to configure an external hostname or IP address immediately raises a `StrictPrivacyViolationError`, preventing network calls across corporate firewalls.

```python
from continuous_legal_memory.adapters.ollama import OllamaAdapter

# Local loopback connection - Allowed
local_adapter = OllamaAdapter(base_url="http://127.0.0.1:11434")

# External cloud connection - Blocked immediately
try:
    cloud_adapter = OllamaAdapter(base_url="https://external-api.provider.com")
except Exception as e:
    print(f"Blocked unauthorized connection: {e}")
```

---

## Offline Unit Testing

To guarantee that CI pipelines run in air-gapped environments without Hugging Face downloads:

- Use `continuous_legal_memory.evaluation.harness.DeterministicTokenEncoder`.
- In pytest suites, inject the `offline_encoder` fixture from `tests/conftest.py`.
