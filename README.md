# Intermediate Project Proposal

## 1. Project Title & Info

**Title:** Bio-AI DTI Query Pipeline: Real-time Drug-Target Interaction Prediction Under Network Constraints
**Name:** Oh Se-jun (오세준) | **Student ID:** 2021270607 | **Team:** Individual

---

## 2. Target Domain & Problem

**Domain:** Health / Pharmaceutical (Drug Discovery)

**Scenario:** A pharmaceutical company runs High-Throughput Screening (HTS) across geographically distributed lab sites. Each site operates an automated compound screening workstation (edge node) that continuously generates DTI queries — drug molecule (SMILES) + target protein (AA sequence) pairs — and transmits them over a corporate WAN to a central AI inference server. Because multiple labs submit simultaneously during peak screening hours, the shared network suffers from congestion-induced latency and packet loss. Additionally, wireless lab environments (disrupted by nearby equipment) cause occasional payload corruption. If the server cannot recover from these disruptions, promising drug candidates are silently dropped from evaluation.

**Problem:** Design a resilient end-to-end ICT pipeline that detects and recovers from real-world network degradation, and still delivers a binding affinity decision for every submitted compound.

---

## 3. System Architecture Diagram

```
[Step 1] Edge Node (Process A)
  Generate DTI queries: {query_id, SMILES, AA sequence, timestamp}
      ↓  ── WAN simulation ──────────────────────────────────────
[Step 2] Transmission Constraints
  Latency: sleep(0.5~2.0 s) | Drop: 15% random | Corrupt: noise σ=0.05
      ↓
[Step 3] Server Node (Process B) — collect & buffer; log drops/corrupts
      ↓
[Step 4] AI Processing & Recovery
  Normal path : Morgan FP + SaProt-650M-4bit → MLP → pKd
  Dropped/corrupt : rolling mean of last 5 valid pKd (imputation)
      ↓
[Step 5] Decision Engine
  pKd ≥ 7.0 → HIGH "Promising" | 5.0–7.0 → MODERATE | < 5.0 → LOW
  packet loss > 30% → ALERT "Network Degraded"
      ↓
[Step 6] Streamlit Dashboard (auto-refresh 2 s)
  pKd time-series | decision badges | packet stats | threshold slider
```
Process A ↔ Process B communicate via `multiprocessing.Queue` (same machine, logically separated).

---

## 4. Intentional Constraints Design

| Constraint | Implementation | Purpose |
|-----------|---------------|---------|
| **Latency** | `time.sleep(random.uniform(0.5, 2.0))` | Simulates congested WAN path |
| **Packet Drop** | `if random.random() < DROP_RATE: skip` (default 15%) | Simulates unreliable delivery; tunable via dashboard slider |
| **Corruption** | Gaussian noise (σ=0.05) injected into Morgan FP vector | Simulates payload bit-flip during transit |

---

## 5. Data & Decision Logic

**Data type:** Text (SMILES strings, amino acid sequences) → derived numerical features (2048-bit Morgan Fingerprint via RDKit; 1280-dim protein embedding via SaProt-650M).
**Source:** DAVIS dataset (30,056 drug-protein pairs, public); KIBA used for cross-validation.

**AI model:** SaProt-650M frozen encoder (NF4 4-bit quantization) + MLP regression head, pre-trained on DAVIS pKd. Validated: Pearson r = 0.7914 (DAVIS), 0.7994 (KIBA).

**Recovery:** Dropped or corrupted packets → rolling mean imputation over the last 5 valid pKd predictions; flagged visually in the dashboard.

**Decision:** pKd threshold rule (HIGH / MODERATE / LOW). Threshold values adjustable in real-time via Streamlit slider. System-level "Network Degraded" alert fires when packet loss rate exceeds 30%.

---

## 6. Tech Stack

| Layer | Technology |
|-------|-----------|
| Language | Python 3.10 |
| AI / ML | PyTorch 2.6, HuggingFace Transformers, bitsandbytes (4-bit) |
| Drug encoding | RDKit (Morgan Fingerprint) |
| Protein encoding | SaProt-650M-AF2 |
| Inter-process comm. | `multiprocessing.Queue` / TCP loopback socket |
| Dashboard | Streamlit |
| Data | DAVIS / KIBA (DeepPurpose library) |
| Version control | GitHub (github.com/ohsejun97/ICT-Application-Technology) |
