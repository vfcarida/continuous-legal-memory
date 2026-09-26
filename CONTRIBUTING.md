# Contributing to Continuous Legal Memory

Welcome, and thank you for your interest in contributing to **Continuous Legal Memory**!

Continuous Legal Memory is an open-source, production-grade memory framework purpose-built for the rule of law, legal AI agents, and regulatory compliance. We welcome contributions from researchers, software engineers, and legal informatics specialists.

---

## Code of Conduct

All contributors and participants are expected to adhere to our [Code of Conduct](CODE_OF_CONDUCT.md) (Contributor Covenant v2.1). Please review it to foster a welcoming and inclusive environment.

---

## Development Environment Setup

### Prerequisites
- Python **3.10**, **3.11**, or **3.12**
- Git

### Initializing the Workspace

1. **Clone the repository**:
   ```bash
   git clone https://github.com/vfcarida/continuous-legal-memory.git
   cd continuous-legal-memory
   ```

2. **Create and activate a virtual environment**:
   ```bash
   python -m venv venv
   # On Linux / macOS:
   source venv/bin/activate
   # On Windows (PowerShell):
   .\venv\Scripts\Activate.ps1
   ```

3. **Install the package in editable mode with development dependencies**:
   ```bash
   pip install --upgrade pip
   pip install -e ".[dev,telemetry]"
   ```

4. **Install pre-commit hooks (recommended)**:
   ```bash
   pre-commit install
   ```

---

## Code Style & Static Analysis

We enforce strict linting, formatting, and typing standards across all modules and tests:

### Ruff (Linter & Formatter)
Run Ruff to check and format the codebase:
```bash
ruff check continuous_legal_memory tests
ruff format continuous_legal_memory tests --check
```

### Mypy (Type Checking)
Ensure type annotations are complete and valid:
```bash
mypy continuous_legal_memory
```

---

## Testing Guidelines

Every code modification must be accompanied by appropriate automated unit or integration tests.

### Running Fast Unit Tests
By default, `pytest` executes all fast, offline-safe unit tests in **under 15 seconds**:
```bash
pytest -v
```

### Offline-First Testing Requirement
**Strict Rule**: Tests must NEVER require an active internet connection or attempt to download external model weights from Hugging Face or public APIs.
- Use the `offline_encoder`, `semantic_mock_encoder`, or `token_encoder` fixtures defined in `tests/conftest.py`.
- Any test that initiates network I/O will fail in offline air-gapped CI environments.

### Running Full Statistical Benchmarks
Empirical benchmarks (multi-seed statistical runs) are separated into a dedicated marker:
```bash
pytest -m benchmark -v
```

---

## Architectural Principles

When proposing changes, adhere to the established architecture:

1. **Hexagonal Architecture (Ports & Adapters)**:
   - Domain models (`continuous_legal_memory/domain/models.py`) and interfaces (`domain/interfaces.py`) must remain decoupled from specific vector database, LLM, or embedding vendor SDKs.
   - External dependencies belong in `continuous_legal_memory/adapters/` or `continuous_legal_memory/storage/`.
2. **Deterministic Precedence**:
   - Legal conflict resolution doctrines (*lex superior*, *lex posterior*, statutory amendment supersession) govern retrieval mathematics.
3. **Multi-Tenant Boundary Enforcement**:
   - Memory records, knowledge graph nodes, and working memory contexts must always respect tenant isolation.
4. **Zero-Trust Auditability**:
   - Prediction results and retrieved context are cryptographically attestable via Ed25519 asymmetric signatures and SHA-256 state hashes.

---

## Pull Request Workflow

1. **Create a topic branch**:
   ```bash
   git checkout -b feature/your-feature-name
   # or
   git checkout -b fix/issue-description
   ```
2. **Commit your changes**:
   - Write clear, descriptive commit messages in imperative English:
     - `feat: add LangChain memory integration adapter`
     - `fix: correct authority rank tiering in precedence resolver`
     - `test: add adversarial multi-tenant graph test`
3. **Verify locally before pushing**:
   ```bash
   ruff check continuous_legal_memory tests
   pytest -v
   ```
4. **Submit your Pull Request**:
   - Clearly explain the problem addressed, the solution implemented, and the tests added.
   - Reference any related GitHub issue numbers.
