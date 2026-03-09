# OGA Replication Package — Optimized Greedy Algorithm for Efficient Test Case Prioritization

This repository contains the replication package for the paper:

**Optimized Greedy Algorithm for Efficient Test Case Prioritization (OGA)**  


Repository link: https://github.com/OGA-algorithm/replication-package

---

## 1) What this replication package provides

- **Algorithms**
  - **OGA** (proposed optimized greedy additional approach)
  - **Baselines** used in the paper
    - GA (Greedy Additional)
    - AGA (Accelerated Greedy Additional)
    - FAST-pw / FAST-all
    - ART-D / ART-F
    - GeTLO (Lexicographical Ordering)

- **Metrics**
  - Prioritization time (execution time)
  - APFD
  - NAPFD
  - PFD@25%

- **Outputs**
  - Per-project prioritized sequence: `SequenceGAMethod_final.txt`
  - Per-project HTML reports (runtime + metrics)
  - Aggregated CSV/HTML summaries used for paper tables/figures

---

## 2) Repository structure (high level)
├── Code/
├── Input_Data/
├── Result/
└── README.md

---

## 3) Dataset folders (`Input_Data/`)

This repository contains multiple prepared datasets under `Input_Data/`:

- `54 project statement coverage/`  
  Statement-level dataset for 54 Java projects.

- `29 project function coverage/`  
  Method/function-level dataset for 29 Java projects.

- `function/`, `original_input/`, `input_adjlist/`  
  Additional dataset variants and intermediate formats used in preprocessing/experiments.

> **Note:** Folder names are preserved as provided in the replication package.  
> When running scripts, set the input path to the desired dataset folder.

---

## 4) Input data format (per project)

Each project folder contains:

### 4.1 Test list
- `testList*`  
  A text file listing test case names (one per line). The line index defines the test ID used by all other files.

### 4.2 Coverage file (choose one level per run)
- **Statement-level:** `state-map.txt` (or `state_map.txt`)
- **Method/function-level:** `function.txt` / `method.txt` (file name may vary by project)

**Format:** one line per test case, adjacency list of covered entity IDs (space-separated integers).  
Example line:0 1 2 10 55 56

Empty line = covers nothing.

### 4.3 Fault matrix
- `mutantKillMatrix*`  
  One row per test. Each row is a 0/1 string where `1` indicates the test kills the mutant at that column.

✅ **Alignment rule (important):**  
Line `i` in `testList`, the coverage file, and `mutantKillMatrix` must refer to the same test.

---

## 5) Environment setup

### 5.1 Create a virtual environment
```bash
python -m venv env
source env/bin/activate
pip install --upgrade pip
pip install numpy pandas matplotlib pympler pyroaring

6) Running the experiments

The scripts in this replication package produce:

SequenceGAMethod_final.txt (the prioritized order)

HTML summaries with execution time and effectiveness metrics

What you need to configure in each script

Most scripts require setting:

root_directory → the input dataset folder (inside Input_Data/)

result_directory → output folder (e.g., inside Result/)

coverage level → statement vs method/function (depending on the script)

Tip: Run statement-level experiments using 54 project statement coverage/
and method-level experiments using 29 project function coverage/.

7) Results (Result/)

The Result/ folder contains generated outputs (HTML summaries and/per-project result folders) used to build the paper’s tables and figures.
