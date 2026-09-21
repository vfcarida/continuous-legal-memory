# Baseline & Preflight Record (CLM-T01)

## 1. Identity & Verification Metadata
- **Task ID**: `CLM-T01`
- **Repository**: `https://github.com/vfcarida/continuous-legal-memory`
- **Target Baseline SHA**: `19e5c5cea7f667472b8dac63cd15e889df8a5fd5`
- **Verified SHA**: `19e5c5cea7f667472b8dac63cd15e889df8a5fd5` (`git rev-parse HEAD`)
- **Working Tree Status**: Clean (`git status --porcelain` returned 0 modified/untracked files)
- **Host Python Environment**: Python 3.12.10, pip 25.0.1
- **Venv Python Environment**: Python 3.10.20, pip 26.2.1 (`venv/`)

### Installed Dependencies (`pip install -e .[dev]`)
- `continuous-legal-memory==0.1.0` (editable)
- `torch==2.14.0`
- `transformers==5.17.0`
- `numpy==2.2.6`
- `typing-extensions==4.16.0`
- `pytest==9.1.1`
- `pytest-cov==7.1.0`
- `ruff==0.16.8`
- `mypy==2.3.1`
- `mlops` optional dependencies: NOT INSTALLED (reserved for CLM-T08 upon approval)

---

## 2. Syntax Compilation (`py_compile`)
- **Command**:
  ```powershell
  & .\venv\Scripts\python.exe -m py_compile $(Get-ChildItem -Path "continuous_legal_memory" -Recurse -Filter "*.py").FullName
  ```
- **Exit Code**: `0`
- **Files Verified**: 25 Python source files under `continuous_legal_memory/` (34 total across repo)
- **Result**: All files compiled without syntax or tokenization errors.

---

## 3. Import Behavior Analysis (Python 3.9 vs 3.10)

### Python 3.10 (Execution Environment)
- **Command**:
  ```powershell
  .\venv\Scripts\python.exe -c "import continuous_legal_memory; print('Import continuous_legal_memory succeeded')"
  .\venv\Scripts\python.exe -c "import continuous_legal_memory.domain.models; print('Import domain.models succeeded')"
  ```
- **Exit Code**: `0`
- **Observed Output**:
  ```text
  Import continuous_legal_memory succeeded
  Import domain.models succeeded
  ```

### Python 3.9 (Known Floor Issue)
- **Target File**: `continuous_legal_memory/domain/exceptions.py:22`
- **Problematic Code**:
  ```python
  def __init__(self, message: str, payload: Any | None = None) -> None:
  ```
- **Failure Cause**:
  In Python 3.9, PEP 604 type union syntax (`Any | None`) is evaluated at class definition time and fails with:
  ```text
  TypeError: unsupported operand type(s) for |: '_SpecialForm' and 'NoneType'
  ```
  because `from __future__ import annotations` is absent.
- **Discrepancy**:
  `pyproject.toml` declares `requires-python = ">=3.9"`, but import fails at runtime on Python 3.9. Fixing this Python floor discrepancy is deferred to `CLM-T02` (per scope boundaries).

---

## 4. Test Execution Record

### Offline-Safe Test Subset (EXECUTED)
- **Command**:
  ```powershell
  .\venv\Scripts\pytest.exe tests/unit/test_domain.py tests/unit/test_ollama_adapter.py tests/unit/test_phase2_memory_tiers.py tests/unit/test_phase3_retrieval_and_mlops.py -v
  ```
- **Exit Code**: `0`
- **Summary**: `13 passed in 63.88s`

| Test File | Test Case | Status | Duration |
| :--- | :--- | :---: | :---: |
| `tests/unit/test_domain.py` | `test_exception_hierarchy` | **PASSED** | < 0.1s |
| `tests/unit/test_domain.py` | `test_memory_record_temporal_validity` | **PASSED** | < 0.1s |
| `tests/unit/test_domain.py` | `test_prediction_result_initialization` | **PASSED** | < 0.1s |
| `tests/unit/test_ollama_adapter.py` | `test_ollama_adapter_strict_privacy_mode` | **PASSED** | < 0.1s |
| `tests/unit/test_ollama_adapter.py` | `test_ollama_adapter_offline_embedding_generation` | **PASSED** | < 0.1s |
| `tests/unit/test_phase2_memory_tiers.py` | `test_working_memory_sliding_window` | **PASSED** | < 0.1s |
| `tests/unit/test_phase2_memory_tiers.py` | `test_episodic_memory_hash_chain_and_decay` | **PASSED** | < 0.1s |
| `tests/unit/test_phase2_memory_tiers.py` | `test_semantic_knowledge_graph_gdpr_and_dependencies` | **PASSED** | < 0.1s |
| `tests/unit/test_phase3_retrieval_and_mlops.py` | `test_bm25_okapi_scoring` | **PASSED** | < 0.1s |
| `tests/unit/test_phase3_retrieval_and_mlops.py` | `test_hybrid_legal_retriever` | **PASSED** | < 0.1s |
| `tests/unit/test_phase3_retrieval_and_mlops.py` | `test_cryptographic_attestation_module` | **PASSED** | < 0.1s |
| `tests/unit/test_phase3_retrieval_and_mlops.py` | `test_agentic_evaluator_metrics` | **PASSED** | < 0.1s |
| `tests/unit/test_phase3_retrieval_and_mlops.py` | `test_telemetry_logger` | **PASSED** | < 0.1s |

---

### Network-Gated Tests (NOT RUN)
The following tests instantiate `LegalMemoryOrchestrator(value_dim=2)` with default parameters, which initializes `HuggingFaceEncoderAdapter` (`continuous_legal_memory/adapters/encoders.py:40-41`). This triggers downloading pre-trained weights for `neuralmind/bert-base-portuguese-cased` (~500MB) from Hugging Face Hub. Under the execution boundary (no model weight downloads, offline-safe baseline), these tests were gated and marked **NOT RUN**:

1. **`tests/unit/test_orchestrator.py`** (5 tests):
   - `test_01_model_weights_are_frozen` — **NOT RUN** (requires BERT download)
   - `test_02_empty_memory_edge_case` — **NOT RUN** (requires BERT download)
   - `test_03_input_validation_and_safety` — **NOT RUN** (requires BERT download)
   - `test_04_surprise_momentum_tracking` — **NOT RUN** (requires BERT download)
   - `test_05_decision_override_and_catastrophic_forgetting` — **NOT RUN** (requires BERT download)

2. **`test_continuous_memory.py`** (5 tests):
   - `TestContinuousLegalMemory.test_01_model_weights_are_frozen` — **NOT RUN** (requires BERT download)
   - `TestContinuousLegalMemory.test_02_empty_memory_edge_case` — **NOT RUN** (requires BERT download)
   - `TestContinuousLegalMemory.test_03_input_validation_and_safety` — **NOT RUN** (requires BERT download)
   - `TestContinuousLegalMemory.test_04_surprise_momentum_tracking` — **NOT RUN** (requires BERT download)
   - `TestContinuousLegalMemory.test_05_decision_override_and_catastrophic_forgetting` — **NOT RUN** (requires BERT download)

---

## 5. Changed-File Inventory & Boundaries
- **Product source code edits (`continuous_legal_memory/`)**: None (0 lines changed).
- **Test files edits (`tests/`, `test_continuous_memory.py`)**: None (0 lines changed).
- **Dependencies or config edits (`pyproject.toml`, CI)**: None (0 lines changed).
- **New Files Created**: `docs/BASELINE.md` (only).
- **Next Task**: `CLM-T02` (Resolve Python floor & PEP 604 type annotation compatibility).
