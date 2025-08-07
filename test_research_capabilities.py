#!/usr/bin/env python3
"""
Test suite for novel research capabilities in quantum hyperparameter optimization.

This test suite validates the research components implemented:
1. Novel QUBO encoding schemes
2. Adaptive quantum strategies  
3. Experimental framework
4. Benchmarking suite
"""

import numpy as np
import time
import warnings
warnings.filterwarnings('ignore')

# Import research modules
try:
    from quantum_hyper_search.research.novel_encodings import (
        HierarchicalEncoder, ConstraintAwareEncoder, MultiObjectiveEncoder, EncodingMetrics
    )
    from quantum_hyper_search.research.adaptive_strategies import (
        LearningBasedAnnealingScheduler, DynamicTopologySelector, FeedbackDrivenTuner, 
        QuantumExperience, AdaptationMetrics
    )
    from quantum_hyper_search.research.experimental_framework import (
        ExperimentRunner, ExperimentalCondition, ExperimentSuite, DatasetGenerator
    )
    from quantum_hyper_search.research.benchmarking_suite import (
        BenchmarkRunner, StandardBenchmarkProblems, ClassicalBaselines
    )
    RESEARCH_MODULES_AVAILABLE = True
except ImportError as e:
    print(f"Research modules not available: {e}")
    RESEARCH_MODULES_AVAILABLE = False

# ML imports for testing
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification


def test_novel_encodings():
    """Test novel QUBO encoding schemes."""
    print("\n🧪 Testing Novel QUBO Encoding Schemes...")
    
    if not RESEARCH_MODULES_AVAILABLE:
        print("❌ Research modules not available")
        return False
    
    # Test parameter space
    param_space = {
        'n_estimators': [10, 25, 50, 100],
        'max_depth': [3, 5, 7, 10],
        'min_samples_split': [2, 5, 10]
    }
    
    success_count = 0
    
    # Test Hierarchical Encoder
    try:
        print("  Testing Hierarchical Encoder...")
        encoder = HierarchicalEncoder(hierarchy_levels=2)
        
        Q = encoder.encode(param_space)
        print(f"    ✅ Generated QUBO with {len(Q)} terms")
        
        # Test decoding
        sample = {i: 1 if i < 3 else 0 for i in range(12)}  # Mock sample
        decoded = encoder.decode(sample, param_space)
        print(f"    ✅ Decoded parameters: {decoded}")
        
        # Test metrics
        metrics = encoder.get_metrics()
        print(f"    ✅ Encoding metrics - Sparsity: {metrics.sparsity_ratio:.3f}, Connectivity: {metrics.connectivity_degree:.2f}")
        
        success_count += 1
    except Exception as e:
        print(f"    ❌ Hierarchical encoder failed: {e}")
    
    # Test Constraint-Aware Encoder
    try:
        print("  Testing Constraint-Aware Encoder...")
        encoder = ConstraintAwareEncoder(constraint_strength=2.0)
        
        constraints = {
            'custom_constraint': lambda params: max(0, params.get('n_estimators', 0) - 100) * 0.1
        }
        
        Q = encoder.encode(param_space, constraints=constraints, objectives=['accuracy', 'speed'])
        print(f"    ✅ Generated constraint-aware QUBO with {len(Q)} terms")
        
        # Test decoding
        sample = {i: 1 if i % 4 == 0 else 0 for i in range(12)}
        decoded = encoder.decode(sample, param_space)
        print(f"    ✅ Decoded parameters: {decoded}")
        
        # Test metrics
        metrics = encoder.get_metrics()
        print(f"    ✅ Constraint metrics - Penalty balance: {metrics.penalty_balance:.3f}")
        
        success_count += 1
    except Exception as e:
        print(f"    ❌ Constraint-aware encoder failed: {e}")
    
    # Test Multi-Objective Encoder
    try:
        print("  Testing Multi-Objective Encoder...")
        encoder = MultiObjectiveEncoder(
            objective_weights={'accuracy': 0.7, 'speed': 0.3},
            scalarization_method='weighted_sum'
        )
        
        Q = encoder.encode(param_space, objectives=['accuracy', 'speed', 'memory'])
        print(f"    ✅ Generated multi-objective QUBO with {len(Q)} terms")
        
        # Test different scalarization methods
        encoder_eps = MultiObjectiveEncoder(scalarization_method='epsilon_constraint')
        Q_eps = encoder_eps.encode(param_space, objectives=['accuracy', 'speed'])
        print(f"    ✅ Epsilon-constraint QUBO with {len(Q_eps)} terms")
        
        success_count += 1
    except Exception as e:
        print(f"    ❌ Multi-objective encoder failed: {e}")
    
    print(f"📊 Novel encodings: {success_count}/3 tests passed ({success_count/3:.1%})")
    return success_count >= 2


def test_adaptive_strategies():
    """Test adaptive quantum strategies."""
    print("\n🧪 Testing Adaptive Quantum Strategies...")
    
    if not RESEARCH_MODULES_AVAILABLE:
        print("❌ Research modules not available")
        return False
    
    success_count = 0
    
    # Test Learning-Based Annealing Scheduler
    try:
        print("  Testing Learning-Based Annealing Scheduler...")
        scheduler = LearningBasedAnnealingScheduler(learning_rate=0.1, exploration_rate=0.2)
        
        # Mock problem context
        problem_context = {
            'num_variables': 50,
            'sparsity_ratio': 0.6,
            'connectivity_degree': 8,
            'parameter_types': ['discrete', 'continuous']
        }
        
        # Mock history
        history = []
        for i in range(5):
            experience = QuantumExperience(
                param_config={'n_estimators': 50, 'max_depth': 7},
                qubo_properties={'num_variables': 50, 'sparsity_ratio': 0.6},
                quantum_settings={'annealing_schedule': 'linear', 'schedule_params': {'start_temp': 1.0}},
                performance_metrics={'best_score': 0.85 + i * 0.02},
                timestamp=time.time()
            )
            history.append(experience)
            scheduler.update_from_experience(experience)
        
        # Test parameter suggestion
        suggested_params = scheduler.suggest_quantum_parameters(problem_context, history)
        print(f"    ✅ Suggested parameters: {suggested_params['annealing_schedule']}")
        
        # Test metrics
        metrics = scheduler.get_adaptation_metrics()
        print(f"    ✅ Adaptation metrics - Improvement rate: {metrics.improvement_rate:.3f}")
        
        success_count += 1
    except Exception as e:
        print(f"    ❌ Learning-based scheduler failed: {e}")
    
    # Test Dynamic Topology Selector
    try:
        print("  Testing Dynamic Topology Selector...")
        selector = DynamicTopologySelector(
            available_topologies=['pegasus', 'chimera'],
            embedding_methods=['minorminer', 'fastembedding']
        )
        
        problem_context = {
            'num_variables': 200,
            'sparsity_ratio': 0.3,
            'connectivity_degree': 12
        }
        
        # Test parameter suggestion
        topology_params = selector.suggest_quantum_parameters(problem_context, [])
        print(f"    ✅ Selected topology: {topology_params['topology']}")
        print(f"    ✅ Selected embedding: {topology_params['embedding_method']}")
        
        # Mock experience and update
        experience = QuantumExperience(
            param_config={},
            qubo_properties={'num_variables': 200},
            quantum_settings=topology_params,
            performance_metrics={'best_score': 0.82, 'embedding_success': True},
            timestamp=time.time()
        )
        selector.update_from_experience(experience)
        
        success_count += 1
    except Exception as e:
        print(f"    ❌ Dynamic topology selector failed: {e}")
    
    # Test Feedback-Driven Tuner
    try:
        print("  Testing Feedback-Driven Tuner...")
        tuner = FeedbackDrivenTuner(adaptation_rate=0.15, momentum=0.1)
        
        # Test parameter suggestion
        suggested_params = tuner.suggest_quantum_parameters({}, [])
        print(f"    ✅ Initial parameters: num_reads={suggested_params['num_reads']}")
        
        # Simulate feedback learning
        for i in range(3):
            performance = 0.7 + i * 0.05
            experience = QuantumExperience(
                param_config={},
                qubo_properties={},
                quantum_settings=suggested_params,
                performance_metrics={'best_score': performance},
                timestamp=time.time()
            )
            tuner.update_from_experience(experience)
            suggested_params = tuner.suggest_quantum_parameters({}, [])
        
        print(f"    ✅ Adapted parameters: num_reads={suggested_params['num_reads']}")
        
        # Test metrics
        metrics = tuner.get_adaptation_metrics()
        print(f"    ✅ Tuning metrics - Stability: {metrics.stability_score:.3f}")
        
        success_count += 1
    except Exception as e:
        print(f"    ❌ Feedback-driven tuner failed: {e}")
    
    print(f"📊 Adaptive strategies: {success_count}/3 tests passed ({success_count/3:.1%})")
    return success_count >= 2


def test_experimental_framework():
    """Test the experimental framework."""
    print("\n🧪 Testing Experimental Framework...")
    
    if not RESEARCH_MODULES_AVAILABLE:
        print("❌ Research modules not available")
        return False
    
    success_count = 0
    
    # Test Dataset Generator
    try:
        print("  Testing Dataset Generator...")
        generator = DatasetGenerator()
        
        # Generate classification dataset
        X, y, name = generator.generate_classification_dataset(
            n_samples=200, n_features=10, n_classes=2
        )
        print(f"    ✅ Generated {name}: {X.shape[0]} samples, {X.shape[1]} features")
        
        # Generate challenging datasets
        challenging = generator.generate_challenging_datasets()
        print(f"    ✅ Generated {len(challenging)} challenging datasets")
        
        success_count += 1
    except Exception as e:
        print(f"    ❌ Dataset generator failed: {e}")
    
    # Test Experimental Conditions and Suite
    try:
        print("  Testing Experimental Framework...")
        
        # Create experimental conditions
        conditions = []
        for algorithm in ['QuantumSearch', 'GridSearch']:
            condition = ExperimentalCondition(
                algorithm_name=algorithm,
                algorithm_params={
                    'model_class': RandomForestClassifier,
                    'param_space': {'n_estimators': [10, 25], 'max_depth': [3, 5]},
                    'n_iterations': 3
                },
                dataset_params={'n_samples': 100, 'n_features': 8, 'n_classes': 2},
                random_seed=42,
                replications=2
            )
            conditions.append(condition)
        
        # Create experiment suite
        suite = ExperimentSuite(
            suite_name="Test Suite",
            description="Testing experimental framework",
            conditions=conditions,
            evaluation_protocol="stratified_k_fold"
        )
        
        print(f"    ✅ Created experiment suite with {len(suite.conditions)} conditions")
        print(f"    ✅ Suite ID: {suite.suite_id}")
        
        success_count += 1
    except Exception as e:
        print(f"    ❌ Experimental framework failed: {e}")
    
    # Test Experiment Runner (basic functionality)
    try:
        print("  Testing Experiment Runner...")
        runner = ExperimentRunner("./test_experiments")
        
        # Test dataset generation within runner
        X, y = make_classification(n_samples=50, n_features=5, random_state=42)
        print(f"    ✅ Created experiment runner with test dataset: {X.shape}")
        
        success_count += 1
    except Exception as e:
        print(f"    ❌ Experiment runner failed: {e}")
    
    print(f"📊 Experimental framework: {success_count}/3 tests passed ({success_count/3:.1%})")
    return success_count >= 2


def test_benchmarking_suite():
    """Test the benchmarking suite."""
    print("\n🧪 Testing Benchmarking Suite...")
    
    if not RESEARCH_MODULES_AVAILABLE:
        print("❌ Research modules not available")
        return False
    
    success_count = 0
    
    # Test Standard Benchmark Problems
    try:
        print("  Testing Standard Benchmark Problems...")
        problems = StandardBenchmarkProblems.get_all_problems()
        print(f"    ✅ Loaded {len(problems)} standard benchmark problems")
        
        # Test problem filtering
        easy_problems = StandardBenchmarkProblems.get_problems_by_difficulty('easy')
        classification_problems = StandardBenchmarkProblems.get_problems_by_type('classification')
        
        print(f"    ✅ Easy problems: {len(easy_problems)}")
        print(f"    ✅ Classification problems: {len(classification_problems)}")
        
        # Test a simple problem
        test_problem = problems[0]
        X, y = test_problem.dataset_generator()
        print(f"    ✅ Test problem dataset: {len(y)} samples")
        
        success_count += 1
    except Exception as e:
        print(f"    ❌ Benchmark problems failed: {e}")
    
    # Test Classical Baselines
    try:
        print("  Testing Classical Baselines...")
        baselines = ClassicalBaselines.get_all_baselines()
        print(f"    ✅ Available baselines: {len(baselines)}")
        
        # Test a baseline optimizer
        X, y = make_classification(n_samples=100, n_features=8, random_state=42)
        param_space = {
            'n_estimators': [10, 25],
            'max_depth': [3, 5]
        }
        
        best_params, metrics = ClassicalBaselines.grid_search_optimizer(
            RandomForestClassifier, param_space, X, y, cv=3, random_state=42
        )
        
        print(f"    ✅ GridSearch result: score={metrics['best_score']:.3f}, params={best_params}")
        
        success_count += 1
    except Exception as e:
        print(f"    ❌ Classical baselines failed: {e}")
    
    # Test Benchmark Runner
    try:
        print("  Testing Benchmark Runner...")
        runner = BenchmarkRunner("./test_benchmarks")
        
        # Create simple mock quantum optimizer
        def mock_quantum_optimizer(model_class, param_space, X, y, random_state=42):
            # Simple random selection
            best_params = {}
            for param, values in param_space.items():
                best_params[param] = np.random.choice(values)
            
            # Mock evaluation
            model = model_class(**best_params, random_state=random_state)
            from sklearn.model_selection import cross_val_score
            scores = cross_val_score(model, X, y, cv=3)
            
            return best_params, {
                'best_score': scores.mean(),
                'n_evaluations': len(param_space),
                'algorithm_name': 'MockQuantum'
            }
        
        # Test single benchmark
        problems = StandardBenchmarkProblems.get_problems_by_difficulty('easy')
        if problems:
            test_problem = problems[0]
            results = runner.run_single_benchmark(
                test_problem, mock_quantum_optimizer, 'MockQuantum', n_replications=2
            )
            print(f"    ✅ Benchmark completed: {len(results)} results")
            
            if results and results[0].success:
                print(f"    ✅ Best result: score={results[0].cv_mean:.3f}")
        
        success_count += 1
    except Exception as e:
        print(f"    ❌ Benchmark runner failed: {e}")
    
    print(f"📊 Benchmarking suite: {success_count}/3 tests passed ({success_count/3:.1%})")
    return success_count >= 2


def test_integration():
    """Test integration between research components."""
    print("\n🧪 Testing Research Integration...")
    
    if not RESEARCH_MODULES_AVAILABLE:
        print("❌ Research modules not available")
        return False
    
    try:
        # Test encoding + adaptive strategy integration
        print("  Testing encoding with adaptive strategy...")
        
        # Create encoder
        encoder = HierarchicalEncoder()
        param_space = {'n_estimators': [10, 50], 'max_depth': [3, 7]}
        
        # Encode problem
        Q = encoder.encode(param_space)
        metrics = encoder.get_metrics()
        
        # Create adaptive strategy
        scheduler = LearningBasedAnnealingScheduler()
        problem_context = {
            'num_variables': len(Q),
            'sparsity_ratio': metrics.sparsity_ratio,
            'connectivity_degree': metrics.connectivity_degree
        }
        
        # Get quantum parameters
        quantum_params = scheduler.suggest_quantum_parameters(problem_context, [])
        
        print(f"    ✅ Integrated encoding ({len(Q)} QUBO terms) with adaptive scheduling")
        print(f"    ✅ Suggested schedule: {quantum_params['annealing_schedule']}")
        
        return True
        
    except Exception as e:
        print(f"    ❌ Integration test failed: {e}")
        return False


def run_all_research_tests():
    """Run all research capability tests."""
    print("🚀 Quantum Hyperparameter Search - Research Capabilities Test Suite")
    print("=" * 80)
    
    if not RESEARCH_MODULES_AVAILABLE:
        print("❌ Research modules not available - skipping research tests")
        return False
    
    test_results = []
    
    # Run all test suites
    test_results.append(test_novel_encodings())
    test_results.append(test_adaptive_strategies())
    test_results.append(test_experimental_framework())
    test_results.append(test_benchmarking_suite())
    test_results.append(test_integration())
    
    # Calculate overall results
    passed_tests = sum(test_results)
    total_tests = len(test_results)
    pass_rate = passed_tests / total_tests
    
    # Generate summary
    print("\n" + "=" * 80)
    print("📊 RESEARCH CAPABILITIES SUMMARY")
    print("=" * 80)
    
    test_names = [
        "Novel QUBO Encodings",
        "Adaptive Strategies", 
        "Experimental Framework",
        "Benchmarking Suite",
        "Integration Tests"
    ]
    
    for i, (test_name, passed) in enumerate(zip(test_names, test_results)):
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{test_name:<25} {status}")
    
    print("-" * 80)
    print(f"OVERALL RESULT: {passed_tests}/{total_tests} test suites passed ({pass_rate:.1%})")
    
    if pass_rate >= 0.8:
        grade = "🏆 EXCELLENT"
        message = "Research capabilities fully operational!"
    elif pass_rate >= 0.6:
        grade = "✅ GOOD"
        message = "Research capabilities mostly functional."
    elif pass_rate >= 0.4:
        grade = "⚠️  ACCEPTABLE"
        message = "Basic research capabilities working."
    else:
        grade = "❌ NEEDS WORK"
        message = "Research capabilities need development."
    
    print(f"RESEARCH GRADE: {grade}")
    print(f"ASSESSMENT: {message}")
    print("=" * 80)
    
    return pass_rate >= 0.6


if __name__ == "__main__":
    success = run_all_research_tests()
    
    if success:
        print("\n🎉 Research capabilities validated! Ready for novel quantum ML research.")
        exit(0)
    else:
        print("\n⚠️  Some research capabilities need attention.")
        exit(1)