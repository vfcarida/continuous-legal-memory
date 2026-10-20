# Scientific, Regulatory & Industry References

This document consolidates the primary peer-reviewed academic literature, international regulatory standards, cryptographic RFCs, and executive governance frameworks underpinning **Continuous Legal Memory (CLM)**.

---

## 1. Academic & Frontier AI Research Foundations

1. **HippoRAG: Neurobiologically Inspired Long-Term Memory for Large Language Models**
   - **Authors:** Bernal Jiménez Gutiérrez, Yiheng Shu, Yu Gu, Michihiro Yasunaga, Yu Su (The Ohio State University / Stanford University)
   - **Venue:** *Advances in Neural Information Processing Systems (NeurIPS)*, 2024 (`arXiv:2405.14831`)
   - **Architectural Anchor:** Formalizes Personalized PageRank (PPR) over knowledge graphs as an associative memory mechanism (`continuous_legal_memory/core/memory_tiers/semantic_memory.py`).

2. **Titans: Learning to Memorize at Test Time**
   - **Authors:** Ali Behrouz, Peilin Zhong, Vahab Mirrokni (Google Research)
   - **Venue:** *Google Research Technical Report*, December 2024 (`arXiv:2501.00663`)
   - **Architectural Anchor:** Surprise-driven dual-timescale memory (`fast_net` / `slow_net`) evaluated in `continuous_legal_memory/experimental/neural_head.py`.

3. **Hallucination-Free? Assessing the Reliability of Leading AI Legal Research Tools**
   - **Authors:** Varun Magesh, Faiz Surani, Matthew Dahl, Mirac Suzgun, Christopher D. Manning, Daniel E. Ho (Stanford RegLab & HAI)
   - **Venue:** *Journal of Empirical Legal Studies*, 2024 (`arXiv:2405.20362`)
   - **Architectural Anchor:** Demonstrates 17%–33% hallucination rates in standard legal RAG systems, motivating CLM's calibrated epistemic abstention (`min_similarity_threshold`) and Split-Conformal prediction sets (`continuous_legal_memory/retrieval/conformal.py`).

4. **LegalBench: A Collaboratively Built Benchmark for Measuring Legal Reasoning in Large Language Models**
   - **Authors:** Neel Guha, Julian Nyarko, Daniel E. Ho, Christopher Ré, et al.
   - **Venue:** *NeurIPS Datasets and Benchmarks Track*, 2023 (`arXiv:2308.11462`)
   - **Architectural Anchor:** Multi-jurisdictional statutory reasoning benchmark tasks (`continuous_legal_memory/evaluation/multijurisdictional.py`).

5. **LegalBench-RAG: A Benchmark for Retrieval-Augmented Generation in the Legal Domain**
   - **Authors:** Nicholas Pipitone, Ghita Houir Alami
   - **Venue:** *arXiv:2408.10343*, 2024
   - **Architectural Anchor:** Snippet-level precision and recall evaluation over legal corpora (`continuous_legal_memory/evaluation/legalbench_eval.py`).

6. **Conformal Risk Control & Split-Conformal Prediction**
   - **Authors:** Vladimir Vovk, Alexander Gammerman, Glenn Shafer (*Algorithmic Learning in a Random World*, Springer, 2005); Anastasios N. Angelopoulos, Stephen Bates, Adam Fisch, Lihua Lei, Tal Schuster (*Conformal Risk Control*, **ICLR 2024**, `arXiv:2208.02814`)
   - **Architectural Anchor:** Finite-sample distribution-free marginal coverage guarantee $P(Y \in \mathcal{C}_\alpha(X)) \ge 1 - \alpha$ (`continuous_legal_memory/retrieval/conformal.py`).

7. **Defeasible Deontic Logic and Normative Conflict Resolution (*Lex Superior, Lex Specialis, Lex Posterior*)**
   - **Authors:** Georg Henrik von Wright (*Deontic Logic*, *Mind*, 1951); Guido Governatori, Antonino Rotolo, Giovanni Sartor (*Handbook of Deontic Logic and Normative Systems* / *Artificial Intelligence and Law*, Springer)
   - **Architectural Anchor:** Tri-canon precedence resolution (`continuous_legal_memory/retrieval/precedence.py`), Standard Deontic Logic ($\mathbf{KD}$) satisfiability (`continuous_legal_memory/reasoning/modal_verifier.py`), and defeasible proof traces (`continuous_legal_memory/reasoning/proof_trace.py`).

8. **Bi-Temporal Database Algebra & Temporal Knowledge Graphs**
   - **Authors:** Richard T. Snodgrass (*Developing Time-Oriented Database Applications in SQL*, Morgan Kaufmann, 1999); Preston Rasmussen et al. (*Zep: A Temporal Knowledge Graph Architecture for Agent Memory*, `arXiv:2501.13956`, 2025)
   - **Architectural Anchor:** Orthogonal valid-time (`at_time`) $\times$ transaction-time (`as_of_transaction_time`) querying (`continuous_legal_memory/domain/models.py`) and regulatory horizon simulation (`continuous_legal_memory/reasoning/temporal_simulator.py`).

9. **Lower-Bounding Term Frequency Normalization (BM25+)**
   - **Authors:** Yuanhua Lv, ChengXiang Zhai
   - **Venue:** *Proceedings of the 20th ACM International Conference on Information and Knowledge Management (CIKM)*, 2011
   - **Architectural Anchor:** Sublinear lexical scoring fused with statutory pin-cite extraction (`continuous_legal_memory/retrieval/hybrid_lexical.py`).

10. **The SMT-LIB Standard (Version 2.6)**
    - **Authors:** Clark Barrett, Pascal Fontaine, Cesare Tinelli
    - **Architectural Anchor:** `QF_UFLIA` formal export bridge for Z3 and cvc5 verification (`continuous_legal_memory/reasoning/smtlib_exporter.py`).

---

## 2. Cryptographic, Security & Interoperability Standards

1. **NIST FIPS 205 (Stateless Hash-Based Digital Signature Standard / SLH-DSA) & RFC 8391 (XMSS / WOTS+):** Post-quantum Winternitz One-Time Signature Plus (`WOTS+-SHA256`) combined with classical `Ed25519` in `continuous_legal_memory/security/pq_hybrid.py`.
2. **RFC 8032 (Edwards-Curve Digital Signature Algorithm — Ed25519):** Zero-trust asymmetric state attestation (`continuous_legal_memory/security/ed25519.py`).
3. **RFC 6962 & RFC 9162 (Certificate Transparency v1/v2):** Domain-separated (`0x00` leaf / `0x01` internal node) 256-bit `SparseMerkleTree` inclusion and non-membership proofs (`continuous_legal_memory/security/merkle.py`) and cross-witness split-view gossip auditing (`continuous_legal_memory/security/timestamp_gossip.py`).
4. **RFC 3161 (Internet X.509 Public Key Infrastructure Time-Stamp Protocol — TSP):** Cryptographic `TimestampToken` binding Merkle state roots and audit log heads (`continuous_legal_memory/security/timestamp_gossip.py`).
5. **Model Context Protocol (MCP — JSON-RPC 2.0) & OpenAPI 3.1.0:** Standardized agent tool/resource transport (`continuous_legal_memory/mcp.py`) and HTTP REST schema discovery (`continuous_legal_memory/server.py`).

---

## 3. International Regulatory & Executive Governance Frameworks

1. **Regulation (EU) 2024/1689 — European Union Artificial Intelligence Act:**
   - **Article 12 (Record-Keeping):** Automatic lifetime provenance logging, Sparse Merkle audit trails, and selective-disclosure proofs.
   - **Article 13 & Article 86 (Transparency & Right to Explanation):** Auditable `DefeasibleProofTrace` derivations.
   - **Article 14 (Human Oversight):** Calibrated epistemic abstention and Split-Conformal prediction sets.
   - **Article 15 (Accuracy, Robustness & Cybersecurity):** Hardware-anchored TEE remote attestation quotes (`AWS_NITRO`, `INTEL_TDX`).
2. **Regulation (EU) 2016/679 — General Data Protection Regulation (GDPR):**
   - **Article 17 (Right to Erasure):** Personal data confinement (`skip_parametric`) and cryptographic non-membership erasure proofs across all memory tiers.
   - **Article 20 & Article 30:** Signed portable compliance bundles (`.clm.bundle.json`).
3. **Private International Law & Global Data Protection Regimes:**
   - **Rome I Regulation (EC) No 593/2008 (Articles 3 & 9):** Party autonomy (*lex voluntatis*) vs. overriding mandatory provisions (*lois de police*) across **EU** (GDPR), **US** (HIPAA / SEC 17a-4 / GLBA), **BR** (LGPD Lei 13.709/2018), **UK** (UK GDPR / DPA 2018), **SG** (PDPA / MAS TRM), **JP** (APPI Art. 171), and **CH** (Swiss FADP / revDSG SR 235.1 & PILA Art. 18).
4. **Gartner AI Trust, Risk and Security Management (AI TRiSM):** Runtime policy enforcement, multi-tenant boundary isolation, cryptographic key rotation (`CryptographicKeyRing`), and OpenMetrics observability.
