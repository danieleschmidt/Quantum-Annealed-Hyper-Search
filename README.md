# Quantum-Annealed-Hyper-Search

Hyperparameter optimization using QUBO (Quadratic Unconstrained Binary Optimization) formulation solved with Simulated Annealing. Provides an Optuna-compatible interface for drop-in HPO integration.

## Install

```bash
pip install numpy scikit-learn
```

## Usage

```python
from qahs.optuna_interface import QAHSOptunaSampler

sampler = QAHSOptunaSampler()
best_params, best_score = sampler.sample(objective_fn, n_trials=20)
```

## How it works

1. **QUBO Encoding** — integer and continuous hyperparameters are encoded as binary variables
2. **Simulated Annealing** — the QUBO problem is approximately solved with SA (Metropolis criterion)
3. **Optuna-compatible API** — drop-in `suggest_int` / `suggest_float` / `sample` interface

## Example: RandomForest HPO

```python
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from qahs.optuna_interface import QAHSOptunaSampler

X, y = make_classification(n_samples=300, n_features=10, random_state=42)

sampler = QAHSOptunaSampler(seed=42)
sampler._get_or_add_param("n_estimators", "int", 10, 100)
sampler._get_or_add_param("max_depth", "int", 2, 10)

def objective(params):
    clf = RandomForestClassifier(
        n_estimators=params["n_estimators"],
        max_depth=params["max_depth"],
        random_state=42
    )
    return -cross_val_score(clf, X, y, cv=3).mean()

best_params, best_score = sampler.sample(objective, n_trials=20)
print(f"Best: {best_params}, accuracy={-best_score:.4f}")
```

## Run demo

```bash
python -m qahs.demo
```

## Run tests

```bash
pip install pytest
pytest tests/ -v
```

## Modules

| Module | Description |
|--------|-------------|
| `qahs.qubo` | `QUBOFormulation` — encodes params as binary vars, builds/decodes QUBO matrix |
| `qahs.sa_solver` | `SimulatedAnnealingSearcher` — SA optimizer with exponential/linear cooling |
| `qahs.optuna_interface` | `QAHSOptunaSampler` — Optuna-compatible HPO sampler |
| `qahs.demo` | `run_demo()` — end-to-end RandomForest example |
