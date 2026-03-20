"""
Demo: Hyperparameter optimization of RandomForestClassifier using QAHS.
"""

from typing import Tuple, Dict


def run_demo() -> Tuple[Dict, float]:
    """
    Demonstrate QAHS by optimizing a RandomForestClassifier on a synthetic dataset.

    Uses QAHSOptunaSampler to search over:
      - n_estimators: 10-100
      - max_depth: 2-10

    Returns:
        (best_params, best_score): Best hyperparameters and their cross-validation score
    """
    from sklearn.datasets import make_classification
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import cross_val_score

    from .optuna_interface import QAHSOptunaSampler

    # Generate a synthetic classification dataset
    X, y = make_classification(
        n_samples=300,
        n_features=10,
        n_informative=5,
        random_state=42,
    )

    # Create sampler and register parameters
    sampler = QAHSOptunaSampler(seed=42)
    sampler._get_or_add_param("n_estimators", "int", 10, 100)
    sampler._get_or_add_param("max_depth", "int", 2, 10)

    def objective(params: dict) -> float:
        """Negative cross-validation accuracy (lower = better for minimization)."""
        n_est = int(params.get("n_estimators", 50))
        depth = int(params.get("max_depth", 5))

        # Clamp to valid range
        n_est = max(1, n_est)
        depth = max(1, depth)

        clf = RandomForestClassifier(
            n_estimators=n_est,
            max_depth=depth,
            random_state=42,
        )
        scores = cross_val_score(clf, X, y, cv=3, scoring="accuracy")
        return -float(scores.mean())  # negate for minimization

    best_params, best_neg_score = sampler.sample(objective, n_trials=20)
    best_score = -best_neg_score  # Convert back to accuracy

    print(f"Best params: {best_params}")
    print(f"Best CV accuracy: {best_score:.4f}")

    return best_params, best_score


if __name__ == "__main__":
    params, score = run_demo()
    print(f"\nResult: {params} -> accuracy={score:.4f}")
