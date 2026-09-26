# Quickstart Guide

This walkthrough demonstrates initializing `LegalMemoryOrchestrator`, ingesting legal rules with conflicting authority ranks, executing predictions, and verifying cryptographic attestation.

---

## 1. Setting up the Orchestrator

```python
from continuous_legal_memory import LegalMemoryOrchestrator
from continuous_legal_memory.adapters.encoders import HuggingFaceEncoderPort

# Initialize with English legal embedding model
orchestrator = LegalMemoryOrchestrator(
    encoder=HuggingFaceEncoderPort(model_name="BAAI/bge-small-en-v1.5"),
    value_dim=2,  # 2-dimensional decision vector: [ALLOW, PROHIBIT]
    engine_mode="structured",
)
```

---

## 2. Ingesting Rules with Precedence

```python
# Baseline internal policy
orchestrator.update_memory(
    "Internal Policy 101: Employee personal devices may connect to enterprise Wi-Fi.",
    action_vector=[1.0, 0.0],  # ALLOW
    authority_rank=1,
)

# Overriding statutory cybersecurity compliance directive
orchestrator.update_memory(
    "Cybersecurity Directive 800: Unmanaged personal devices are strictly prohibited on enterprise networks.",
    action_vector=[0.0, 1.0],  # PROHIBIT
    authority_rank=6,  # Higher rank overrides rank 1
)
```

---

## 3. Querying & Verifying Precedence

```python
result = orchestrator.predict("Can an employee use their personal phone on company Wi-Fi?")

print("Predicted Action:", result.predicted_action_vector)
# Resolves to [0.0, 1.0] (PROHIBIT)
print("Governing Rule:", result.most_relevant_rule)
# 'Cybersecurity Directive 800: Unmanaged personal devices are strictly prohibited on enterprise networks.'
```

---

## 4. Generating & Verifying Cryptographic Attestation

```python
from continuous_legal_memory.security import AsymmetricAttestationModule

# Sign result with Ed25519
attestation_module = AsymmetricAttestationModule()
token = attestation_module.sign_attestation(result)

# Independent verification using only public key
auditor = AsymmetricAttestationModule(public_key=attestation_module.public_key_hex)
is_authentic = auditor.verify_attestation(token, result)
print(f"Audit Proof Valid: {is_authentic}")
```
