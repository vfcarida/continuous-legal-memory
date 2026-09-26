# Security Policy

## Supported Versions

We prioritize security patches and vulnerability mitigations for the following active releases:

| Version | Supported          | Security Maintenance Status |
| :---    | :---:              | :---                        |
| 0.1.x   | :white_check_mark: | Active Development & Patches|
| < 0.1.0 | :x:                | Deprecated / Not Supported  |

---

## Reporting a Vulnerability

The Continuous Legal Memory team takes the security, confidentiality, and integrity of legal intelligence systems seriously. If you discover a security vulnerability or potential privacy breach, please report it privately.

**DO NOT file a public GitHub issue for security vulnerabilities.**

### Reporting Procedure

1. **Email**: Send your findings directly to the lead maintainer at **`vfcarida@gmail.com`**.
2. **Subject**: Use `[SECURITY REPORT] Continuous Legal Memory - <Brief Vulnerability Description>`.
3. **Report Contents**:
   - Detailed description of the vulnerability.
   - Proof of Concept (PoC) script or minimal reproducible example.
   - Potential impact on tenant isolation, cryptographic attestation, or sensitive legal data.
   - Any recommended remediation or patches.

### Response Timeline

- **Acknowledgment**: Within 48 hours of receipt.
- **Triage & Assessment**: Within 5 business days.
- **Remediation & Advisory**: Target patch release within 14 calendar days of confirmation.

We kindly request that you observe **Responsible Disclosure** guidelines and allow sufficient time for remediation before publishing any security advisories.

---

## Security Architecture & Scope

Continuous Legal Memory is designed for regulated enterprise legal environments with attorney-client confidentiality and strict compliance requirements:

1. **Multi-Tenant Isolation**:
   - All memory tiers (`WorkingMemory`, `SemanticKnowledgeGraph`, `SqliteMemoryStore`) enforce tenant scoping.
   - Cross-tenant retrieval, edge linkage, and rule inheritance are strictly prohibited.
2. **Cryptographic Attestation & Audit Trails**:
   - Attestation tokens provide tamper-evident proofs using RFC 8032 **Ed25519** digital signatures and **HMAC-SHA256** digests.
   - Verification must fail if canonical payloads, decision vectors, or timestamps are altered.
3. **Data Protection & Offline Execution**:
   - Adapters (e.g. `OllamaAdapter`) strictly enforce loopback constraints (`127.0.0.1` / `localhost`) to prevent unauthorized outbound exfiltration of privileged legal texts.
4. **GDPR Article 17 Right to Erasure**:
   - Erasure operations must purge records across working, episodic, and semantic tiers, leaving irreversible SHA-256 cryptographic tombstones.
5. **Adversarial & Poisoning Protection**:
   - The memory engine enforces input vector dimensionality validation, text sanitization, and contradiction detection to mitigate prompt injection and retrieval poisoning.
