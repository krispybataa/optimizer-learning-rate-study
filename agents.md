# Agentic Development Environment

This document defines the roles, standards, and context for AI agents working on the **Optimizer and Learning Rate Study of Ovarian Cancer Prediction** repository.

## Project Context
- **Objective:** Systematically evaluate deep learning configurations for ovarian cancer classification.
- **Matrix:** 84 runs (4 architectures × 7 optimizers × 3 learning rates).
- **Stack:** TensorFlow/Keras, Python 3.10+, LaTeX (for the final paper).
- **Data:** STRAMPN Histopathological Images (987 images).

## Agent Roles

### 1. The Research Agent (Antigravity)
- **Primary Goal:** Analyze experiment results and maintain the LaTeX manuscript.
- **Responsibilities:**
  - Verify result integrity across CSVs, plots, and LaTeX tables.
  - Update `docs/latex/` sections with new findings.
  - Ensure all source code appendices stay synchronized with `src/`.
  - Resolve LaTeX compilation issues (pdflatex/bibtex).

### 2. The Simulation Agent
- **Primary Goal:** Execute and monitor the 84-run training pipeline.
- **Responsibilities:**
  - Maintain `src/run_all.py` and `src/train.py`.
  - Handle GPU/memory constraints during large-scale runs.
  - Generate aggregate metrics in `results/summary/`.

### 3. The Visualization Agent
- **Primary Goal:** Generate publication-quality figures and heatmaps.
- **Responsibilities:**
  - Maintain `notebooks/03_results_analysis.ipynb`.
  - Ensure heatmaps in `docs/latex/figures/` are up-to-date.
  - Maintain TikZ diagrams in `docs/latex/sections/04_pipeline.tex`.

## Standards for AI Agents

### 1. Code Integrity
- Always use the `_clean.py` variants for inclusion in LaTeX appendices. These files must be free of U+2014 (em-dash) and other non-ASCII characters that break pdflatex.
- Keep `src/` modular: `model_builder.py` for architecture, `train.py` for loops, `evaluate.py` for metrics.

### 2. LaTeX Best Practices
- Use `\label{sec:...}` for sections and `\label{tab:...}` for tables.
- Use `\ref{...}` for cross-references.
- When adding results, ensure `longtable` environments are used for tables spanning multiple pages.
- Wrap longtables in `\begingroup\setstretch{1}...` to avoid spacing issues.

### 3. File Naming
- Result metrics: `results/metrics/<run_name>.csv`
- Result curves: `results/curves/<run_name>_acc_loss.png`
- Master results: `results/summary/master_results.csv`

## IDE-Specific Rules
- **Cursor:** Refer to `.cursorrules` for specific prompting instructions.
- **Cline:** Refer to `.clinerules` for custom instructions.
