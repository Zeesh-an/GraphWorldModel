# Graph World Model

Research codebase for **Graph Inverse Problems** — learning to solve combinatorial
optimization and inference tasks on graphs using ML/deep learning.

The core idea: rather than running expensive classical algorithms at inference time,
train models that learn the structure of solutions from graph data and generalize
across problem instances.

---

## Tasks

| Task | Folder | Status |
|------|--------|--------|
| Influence Maximization | `World_model/` | Active |
| *(more tasks to be added)* | — | Planned |

---

## Project Structure

```
GraphWorldModel/
├── README.md                 # this file — project overview + setup
├── RESEARCH.md               # literature survey (methods, datasets, SOTA)
├── requirements.txt          # Python dependencies
│
├── Data/                     # data generation pipelines
│   └── README.md             # how to generate data per task
│
├── World_model/              # model implementations
│   └── README.md             # architecture and training details
│
└── Baselines/                # reference implementations from prior work
    └── README.md             # list of baselines and how to run them
```

---

## Environment Setup

Requires **Python 3.12**.

```bash
# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

**Dependencies:** `torch`, `numpy`, `scipy`, `networkx`, `pandas`, `scikit-learn`

---

## Quick Start

Each task has its own data generation and training pipeline.
See the README in the relevant subfolder:

- [Data/README.md](Data/README.md) — generate training data
- [World_model/README.md](World_model/README.md) — train a model
- [Baselines/README.md](Baselines/README.md) — run baseline comparisons
