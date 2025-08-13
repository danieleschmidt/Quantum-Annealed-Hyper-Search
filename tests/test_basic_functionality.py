"""
Basic functionality tests for the quantum hyperparameter search library.
"""

import pytest
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification

from quantum_hyper_search import QuantumHyperSearch


@pytest.fixture
def sample_data():
    """Generate sample classification data for testing."""
    X, y = make_classification(
        n_samples=100,
        n_features=10,
        n_classes=2,
        random_state=42
    )
    return X, y


@pytest.fixture 
def sample_param_space():
    """Define a simple parameter space for testing."""
    return {
        'n_estimators': [10, 50, 100],
        'max_depth': [3, 5, 7],
        'min_samples_split': [2, 5, 10]
    }


def test_quantum_hyper_search_initialization():
    """Test basic initialization of QuantumHyperSearch."""
    qhs = QuantumHyperSearch(backend='simple')
    assert qhs is not None
    assert qhs.backend_name == 'simple'


def test_quantum_hyper_search_optimization(sample_data, sample_param_space):
    """Test the basic optimization workflow."""
    X, y = sample_data
    
    qhs = QuantumHyperSearch(backend='simple')
    
    best_params, history = qhs.optimize(
        model_class=RandomForestClassifier,
        param_space=sample_param_space,
        X=X,
        y=y,
        n_iterations=3,
        quantum_reads=10,
        cv_folds=3,
        scoring='accuracy'
    )
    
    # Verify results
    assert best_params is not None
    assert isinstance(best_params, dict)
    assert history is not None
    assert history.best_score > 0
    assert len(history.trials) > 0
    
    # Verify parameter values are from the search space
    for param_name, param_value in best_params.items():
        assert param_name in sample_param_space
        assert param_value in sample_param_space[param_name]


def test_empty_parameter_space():
    """Test handling of empty parameter space."""
    qhs = QuantumHyperSearch(backend='simple')
    
    with pytest.raises(Exception):
        X, y = make_classification(n_samples=50, n_features=5, random_state=42)
        qhs.optimize(
            model_class=RandomForestClassifier,
            param_space={},
            X=X,
            y=y,
            n_iterations=1
        )


def test_invalid_model_class():
    """Test handling of invalid model class."""
    qhs = QuantumHyperSearch(backend='simple')
    
    with pytest.raises(Exception):
        X, y = make_classification(n_samples=50, n_features=5, random_state=42)
        qhs.optimize(
            model_class="not_a_class",
            param_space={'n_estimators': [10, 20]},
            X=X,
            y=y,
            n_iterations=1
        )


if __name__ == "__main__":
    pytest.main([__file__])