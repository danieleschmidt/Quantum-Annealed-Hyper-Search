"""
Setup script for Quantum Annealed Hyperparameter Search.
"""

from setuptools import setup, find_packages
import os

# Read README for long description
readme_path = os.path.join(os.path.dirname(__file__), 'README.md')
with open(readme_path, 'r', encoding='utf-8') as f:
    long_description = f.read()

# Read version from package safely
import re

def get_version():
    """Safely extract version from __init__.py"""
    version_path = os.path.join(os.path.dirname(__file__), 'quantum_hyper_search', '__init__.py')
    with open(version_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Extract version using regex instead of exec
    version_match = re.search(r'^__version__\s*=\s*[\'"]([^\'"]*)[\'"]', content, re.MULTILINE)
    author_match = re.search(r'^__author__\s*=\s*[\'"]([^\'"]*)[\'"]', content, re.MULTILINE)
    email_match = re.search(r'^__email__\s*=\s*[\'"]([^\'"]*)[\'"]', content, re.MULTILINE)
    
    if not version_match:
        raise RuntimeError("Unable to find version string.")
    
    return {
        '__version__': version_match.group(1),
        '__author__': author_match.group(1) if author_match else "Daniel Schmidt",
        '__email__': email_match.group(1) if email_match else "daniel@example.com"
    }

version = get_version()

setup(
    name="quantum-annealed-hyper-search",
    version=version['__version__'],
    author=version['__author__'],
    author_email=version['__email__'],
    description="Hybrid quantum-classical library for hyperparameter optimization using D-Wave quantum annealers",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/danieleschmidt/quantum-annealed-hyper-search",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.21.0",
        "scipy>=1.7.0",
        "pandas>=1.3.0",
        "scikit-learn>=1.0.0",
        "matplotlib>=3.5.0",
        "seaborn>=0.11.0",
        "tqdm>=4.62.0",
        "pydantic>=1.8.0",
        "typing-extensions>=4.0.0",
        "psutil>=5.8.0",
    ],
    extras_require={
        "dwave": [
            "dwave-ocean-sdk>=6.0.0",
            "dimod>=0.12.0",
            "dwave-system>=1.19.0",
            "dwave-hybrid>=0.6.0",
            "minorminer>=0.2.0",
        ],
        "simulators": [
            "neal>=0.5.9",
            "tabu>=0.1.0",
        ],
        "optimizers": [
            "optuna>=3.0.0",
            "ray[tune]>=2.0.0",
            "hyperopt>=0.2.7",
            "bayesian-optimization>=1.4.0",
        ],
        "ml": [
            "torch>=1.12.0",
            "tensorflow>=2.9.0",
            "xgboost>=1.6.0",
            "lightgbm>=3.3.0",
        ],
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=22.0.0",
            "isort>=5.10.0",
            "flake8>=5.0.0",
            "mypy>=0.990",
            "pre-commit>=2.20.0",
            "sphinx>=5.0.0",
            "sphinx-rtd-theme>=1.0.0",
            "jupyter>=1.0.0",
        ],
        "all": [
            "quantum-annealed-hyper-search[dwave,simulators,optimizers,ml]",
        ],
    },
    entry_points={
        "console_scripts": [
            "quantum-hyper-search=quantum_hyper_search.cli:main",
        ],
    },
    project_urls={
        "Bug Reports": "https://github.com/danieleschmidt/quantum-annealed-hyper-search/issues",
        "Source": "https://github.com/danieleschmidt/quantum-annealed-hyper-search",
        "Documentation": "https://quantum-hyper-search.readthedocs.io",
    },
    keywords="quantum machine-learning hyperparameter-optimization d-wave annealing",
    zip_safe=False,
    include_package_data=True,
)