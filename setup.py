#!/usr/bin/env python3
"""
Quantum Hyperparameter Search - Production Setup
Enterprise-grade quantum-classical hyperparameter optimization framework.
"""

from setuptools import setup, find_packages
import os

# Read requirements
def read_requirements(filename):
    """Read requirements from file."""
    requirements_path = os.path.join(os.path.dirname(__file__), filename)
    if os.path.exists(requirements_path):
        with open(requirements_path, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f if line.strip() and not line.startswith('#')]
    return []

# Read README for long description
def read_readme():
    """Read README for long description."""
    readme_path = os.path.join(os.path.dirname(__file__), 'README.md')
    if os.path.exists(readme_path):
        with open(readme_path, 'r', encoding='utf-8') as f:
            return f.read()
    return "Quantum-enhanced hyperparameter optimization framework"

# Package metadata
setup(
    name="quantum-hyper-search",
    version="1.0.0",
    author="Terragon Labs",
    author_email="contact@terragonlabs.com",
    description="Enterprise-grade quantum-classical hyperparameter optimization framework",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/terragon-labs/quantum-hyper-search",
    project_urls={
        "Bug Tracker": "https://github.com/terragon-labs/quantum-hyper-search/issues",
        "Documentation": "https://quantum-hyper-search.terragonlabs.com/docs",
        "Source Code": "https://github.com/terragon-labs/quantum-hyper-search",
        "Homepage": "https://terragonlabs.com",
    },
    
    # Package discovery
    packages=find_packages(exclude=["tests*", "docs*", "examples*"]),
    include_package_data=True,
    
    # Python version requirement
    python_requires=">=3.8",
    
    # Dependencies
    install_requires=[
        # Core dependencies
        "numpy>=1.21.0",
        "scikit-learn>=1.0.0",
        "pandas>=1.3.0",
        "scipy>=1.7.0",
        
        # Quantum computing (optional but recommended)
        "dwave-ocean-sdk>=6.0.0",
        "dwave-system>=1.18.0",
        "dwave-neal>=0.6.0",
        "dimod>=0.12.0",
        
        # Machine learning enhancement
        "xgboost>=1.5.0",
        "lightgbm>=3.3.0",
        
        # Optimization and numerical
        "optuna>=3.0.0",
        "hyperopt>=0.2.7",
        "bayesian-optimization>=1.4.0",
        
        # Enterprise features
        "cryptography>=3.4.0",
        "pyjwt>=2.4.0",
        "psutil>=5.8.0",
        
        # Monitoring and observability
        "prometheus-client>=0.14.0",
        "plotly>=5.0.0",
        "matplotlib>=3.5.0",
        "seaborn>=0.11.0",
        
        # Development and testing
        "tqdm>=4.62.0",
        "colorlog>=6.6.0",
        "click>=8.0.0",
        
        # Parallel processing
        "joblib>=1.1.0",
        "ray[default]>=2.0.0",
        
        # Data serialization
        "cloudpickle>=2.0.0",
        "h5py>=3.6.0",
    ],
    
    # Optional dependencies
    extras_require={
        "quantum": [
            "dwave-ocean-sdk>=6.0.0",
            "dwave-system>=1.18.0",
            "dwave-neal>=0.6.0",
            "qiskit>=0.39.0",
            "cirq>=1.0.0",
        ],
        "gpu": [
            "cupy-cuda11x>=10.0.0",
            "tensorflow-gpu>=2.10.0",
            "torch>=1.12.0",
        ],
        "enterprise": [
            "cryptography>=3.4.0",
            "pyjwt>=2.4.0",
            "prometheus-client>=0.14.0",
            "grafana-api>=1.0.3",
        ],
        "research": [
            "jupyter>=1.0.0",
            "notebook>=6.4.0",
            "ipywidgets>=7.6.0",
            "sympy>=1.9.0",
        ],
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "pytest-benchmark>=4.0.0",
            "black>=22.0.0",
            "flake8>=5.0.0",
            "mypy>=0.991",
            "sphinx>=5.0.0",
            "sphinx-rtd-theme>=1.0.0",
        ],
        "all": [
            # Includes all optional dependencies
            "dwave-ocean-sdk>=6.0.0",
            "qiskit>=0.39.0",
            "cupy-cuda11x>=10.0.0",
            "tensorflow-gpu>=2.10.0",
            "cryptography>=3.4.0",
            "prometheus-client>=0.14.0",
            "jupyter>=1.0.0",
            "pytest>=7.0.0",
        ]
    },
    
    # Entry points for command-line tools
    entry_points={
        "console_scripts": [
            "qhs-optimize=quantum_hyper_search.cli.optimize:main",
            "qhs-benchmark=quantum_hyper_search.cli.benchmark:main",
            "qhs-monitor=quantum_hyper_search.cli.monitor:main",
            "qhs-server=quantum_hyper_search.server.api:main",
        ],
    },
    
    # Package data
    package_data={
        "quantum_hyper_search": [
            "config/*.yaml",
            "config/*.json",
            "templates/*.html",
            "static/css/*.css",
            "static/js/*.js",
        ],
    },
    
    # Classification
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: System :: Distributed Computing",
    ],
    
    # Keywords for PyPI search
    keywords=[
        "quantum computing",
        "hyperparameter optimization",
        "machine learning",
        "quantum annealing",
        "automl",
        "bayesian optimization",
        "enterprise ml",
        "distributed computing",
        "optimization",
        "quantum machine learning"
    ],
    
    # License
    license="Apache License 2.0",
    
    # Zip safe
    zip_safe=False,
    
    # Additional metadata
    platforms=["any"],
    
    # Minimum setuptools version
    setup_requires=["setuptools>=45", "wheel>=0.37.0"],
)