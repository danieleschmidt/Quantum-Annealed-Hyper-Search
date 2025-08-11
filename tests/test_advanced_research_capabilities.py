"""
Comprehensive Test Suite for Advanced Research Capabilities

Test suite covering all novel quantum algorithms and research features
including parallel tempering, error correction, and quantum walks.
"""

import unittest
import numpy as np
import time
from unittest.mock import Mock, patch
from dataclasses import dataclass
from typing import Dict, Any, List

# Import the research modules
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from quantum_hyper_search.research.quantum_parallel_tempering import (
    QuantumParallelTempering, TemperingParams, TemperingResults
)
from quantum_hyper_search.research.quantum_error_correction import (
    QuantumErrorCorrection, ErrorCorrectionParams, CorrectionResults
)
from quantum_hyper_search.research.quantum_walk_optimizer import (
    QuantumWalkOptimizer, QuantumWalkParams, WalkResults
)
from quantum_hyper_search.research.quantum_bayesian_optimization import (
    QuantumBayesianOptimizer, BayesianOptParams, BayesianResults
)
from quantum_hyper_search.optimization.distributed_quantum_optimization import (
    DistributedQuantumOptimizer, OptimizationTask, TaskPriority
)
from quantum_hyper_search.optimization.adaptive_resource_management import (
    AdaptiveResourceManager, ResourceRequest, AllocationStrategy
)

class MockQuantumBackend:
    """Mock quantum backend for testing"""
    
    def __init__(self):
        self.sample_count = 0
    
    def sample_qubo(self, Q, **kwargs):
        """Mock QUBO sampling"""
        self.sample_count += 1
        n = Q.shape[0]
        
        # Generate mock samples
        samples = []
        for _ in range(kwargs.get('num_reads', 10)):
            sample = {i: np.random.choice([0, 1]) for i in range(n)}
            energy = float(np.random.uniform(-1, 1))
            
            mock_sample = Mock()
            mock_sample.sample = sample
            mock_sample.energy = energy
            mock_sample.num_occurrences = 1
            
            samples.append(mock_sample)
        
        # Mock result object
        mock_result = Mock()
        mock_result.data.return_value = samples
        
        return mock_result

class TestQuantumParallelTempering(unittest.TestCase):
    """Test quantum parallel tempering implementation"""
    
    def setUp(self):
        self.backend = MockQuantumBackend()
        self.tempering_params = TemperingParams(
            temperatures=[0.1, 0.5, 1.0, 2.0],
            exchange_attempts=10,
            cooling_schedule="exponential"
        )
        self.optimizer = QuantumParallelTempering(
            backend=self.backend,
            tempering_params=self.tempering_params
        )
    
    def test_initialization(self):
        """Test proper initialization"""
        self.assertEqual(len(self.optimizer.params.temperatures), 4)
        self.assertEqual(self.optimizer.params.exchange_attempts, 10)
        self.assertTrue(self.optimizer.enable_quantum_tunneling)
    
    def test_replica_initialization(self):
        """Test replica initialization"""
        qubo_matrix = np.random.random((5, 5))
        qubo_matrix = (qubo_matrix + qubo_matrix.T) / 2  # Make symmetric
        
        replicas = self.optimizer._initialize_replicas(qubo_matrix)
        
        self.assertEqual(len(replicas), 4)  # One per temperature
        for replica in replicas:
            self.assertIn('state', replica)
            self.assertIn('temperature', replica)
            self.assertIn('energy', replica)
            self.assertEqual(len(replica['state']), 5)
    
    def test_optimization_execution(self):
        """Test full optimization execution"""
        qubo_matrix = np.array([
            [1, -0.5],
            [-0.5, 1]
        ])
        
        results = self.optimizer.optimize(
            qubo_matrix=qubo_matrix,
            max_iterations=20,
            convergence_threshold=1e-3
        )
        
        self.assertIsInstance(results, TemperingResults)
        self.assertIsNotNone(results.best_solution)
        self.assertIsInstance(results.best_energy, float)
        self.assertIsInstance(results.quantum_advantage_achieved, bool)
        self.assertGreater(results.convergence_time, 0)
    
    def test_energy_calculation(self):
        """Test QUBO energy calculation"""
        qubo_matrix = np.array([[1, -0.5], [-0.5, 1]])
        state = np.array([1, 0])
        
        energy = self.optimizer._calculate_energy(state, qubo_matrix)
        expected = state.T @ qubo_matrix @ state
        
        self.assertAlmostEqual(energy, expected, places=6)
    
    def test_quantum_enhancement(self):
        """Test quantum enhancement features"""
        # Test with quantum enhancement enabled
        qubo_matrix = np.random.random((3, 3))
        results_quantum = self.optimizer.optimize(qubo_matrix, max_iterations=10)
        
        # Test without quantum enhancement
        self.optimizer.enable_quantum_tunneling = False
        results_classical = self.optimizer.optimize(qubo_matrix, max_iterations=10)
        
        # Both should complete successfully
        self.assertIsInstance(results_quantum.best_energy, float)
        self.assertIsInstance(results_classical.best_energy, float)

class TestQuantumErrorCorrection(unittest.TestCase):
    """Test quantum error correction implementation"""
    
    def setUp(self):
        self.backend = MockQuantumBackend()
        self.correction_params = ErrorCorrectionParams(
            repetition_code_distance=3,
            majority_voting_threshold=0.6
        )
        self.corrector = QuantumErrorCorrection(
            backend=self.backend,
            correction_params=self.correction_params
        )
    
    def test_initialization(self):
        """Test proper initialization"""
        self.assertEqual(self.corrector.params.repetition_code_distance, 3)
        self.assertEqual(self.corrector.params.majority_voting_threshold, 0.6)
        self.assertTrue(self.corrector.enable_adaptive_correction)
    
    def test_solution_generation(self):
        """Test initial solution generation"""
        qubo_matrix = np.random.random((4, 4))
        
        solution = self.corrector._generate_initial_solution(qubo_matrix)
        
        self.assertIn('variables', solution)
        self.assertIn('energy', solution)
        self.assertEqual(len(solution['variables']), 4)
    
    def test_repetition_code(self):
        """Test repetition code application"""
        solution = {
            'variables': {'0': 1, '1': 0, '2': 1},
            'energy': 0.5,
            'num_occurrences': 1
        }
        qubo_matrix = np.eye(3)
        
        encoded_solutions = self.corrector._apply_repetition_code(qubo_matrix, solution)
        
        self.assertEqual(len(encoded_solutions), 3)  # Distance 3
        for encoded in encoded_solutions:
            self.assertIn('variables', encoded)
            self.assertIn('energy', encoded)
    
    def test_error_detection_and_correction(self):
        """Test error detection and correction"""
        # Create solutions with intentional disagreements
        solutions = [
            {'variables': {'0': 1, '1': 0}, 'energy': 0.1, 'num_occurrences': 1},
            {'variables': {'0': 1, '1': 1}, 'energy': 0.2, 'num_occurrences': 1},
            {'variables': {'0': 0, '1': 0}, 'energy': 0.3, 'num_occurrences': 1}
        ]
        qubo_matrix = np.eye(2)
        
        corrected_solution, corrections = self.corrector._detect_and_correct_errors(
            solutions, qubo_matrix
        )
        
        self.assertIn('variables', corrected_solution)
        self.assertIn('energy', corrected_solution)
        self.assertIsInstance(corrections, int)
    
    def test_full_correction_process(self):
        """Test complete error correction process"""
        qubo_matrix = np.array([[1, -0.5], [-0.5, 1]])
        
        results = self.corrector.correct_qubo_solution(
            qubo_matrix=qubo_matrix,
            num_correction_rounds=3
        )
        
        self.assertIsInstance(results, CorrectionResults)
        self.assertIsNotNone(results.original_solution)
        self.assertIsNotNone(results.corrected_solution)
        self.assertIsInstance(results.error_rate, float)
        self.assertIsInstance(results.confidence_score, float)

class TestQuantumWalkOptimizer(unittest.TestCase):
    """Test quantum walk optimization implementation"""
    
    def setUp(self):
        self.backend = MockQuantumBackend()
        self.walk_params = QuantumWalkParams(
            walk_length=50,
            mixing_angle=np.pi/4
        )
        self.optimizer = QuantumWalkOptimizer(
            backend=self.backend,
            walk_params=self.walk_params
        )
    
    def test_initialization(self):
        """Test proper initialization"""
        self.assertEqual(self.optimizer.params.walk_length, 50)
        self.assertAlmostEqual(self.optimizer.params.mixing_angle, np.pi/4)
        self.assertTrue(self.optimizer.enable_entanglement)
    
    def test_objective_function_optimization(self):
        """Test optimization with objective function"""
        def simple_objective(x):
            # Simple quadratic function
            return sum((x[i] - 0.5) ** 2 for i in range(len(x)))
        
        results = self.optimizer.optimize(
            objective_function=simple_objective,
            search_space_dim=3,
            max_iterations=20
        )
        
        self.assertIsInstance(results, WalkResults)
        self.assertIn('variables', results.best_solution)
        self.assertIsInstance(results.best_energy, float)
        self.assertIsInstance(results.exploration_coverage, float)
        self.assertGreater(results.convergence_steps, 0)
    
    def test_adaptive_mixing_angle(self):
        """Test adaptive mixing angle calculation"""
        # Test angle adaptation over time
        angles = []
        for step in range(0, 100, 10):
            angle = self.optimizer._adaptive_mixing_angle(step)
            angles.append(angle)
        
        # Should generally decrease over time
        self.assertGreater(angles[0], angles[-1])
        
        # All angles should be in valid range
        for angle in angles:
            self.assertGreaterEqual(angle, 0.1)
            self.assertLessEqual(angle, np.pi/2)
    
    def test_entanglement_boost(self):
        """Test quantum entanglement boost feature"""
        # Mock quantum walker
        from quantum_hyper_search.research.quantum_walk_optimizer import QuantumWalker
        walker = QuantumWalker(
            dimension=3,
            initial_position=np.array([1, 0, 1]),
            coin_params=self.optimizer.params.coin_parameters
        )
        
        visited_states = {(1, 0, 0), (0, 1, 1), (1, 1, 0)}
        
        def mock_objective(x):
            return sum(x)
        
        entangled_positions = self.optimizer._apply_entanglement_boost(
            walker, visited_states, mock_objective
        )
        
        self.assertIsInstance(entangled_positions, list)
        for pos in entangled_positions:
            self.assertEqual(len(pos), 3)  # Same dimension

class TestQuantumBayesianOptimization(unittest.TestCase):
    """Test quantum Bayesian optimization implementation"""
    
    def setUp(self):
        self.backend = MockQuantumBackend()
        self.bayes_params = BayesianOptParams(
            acquisition_function="quantum_expected_improvement",
            max_evaluations=20
        )
        self.optimizer = QuantumBayesianOptimizer(
            backend=self.backend,
            bayes_params=self.bayes_params
        )
    
    def test_initialization(self):
        """Test proper initialization"""
        self.assertEqual(self.optimizer.params.acquisition_function, "quantum_expected_improvement")
        self.assertEqual(self.optimizer.params.max_evaluations, 20)
        self.assertTrue(self.optimizer.enable_quantum_kernel)
    
    def test_optimization_execution(self):
        """Test full Bayesian optimization"""
        def objective_function(params):
            return params['x'] ** 2 + params['y'] ** 2
        
        parameter_bounds = {
            'x': (-2.0, 2.0),
            'y': (-2.0, 2.0)
        }
        
        results = self.optimizer.optimize(
            objective_function=objective_function,
            parameter_bounds=parameter_bounds,
            n_initial_points=3
        )
        
        self.assertIsInstance(results, BayesianResults)
        self.assertIn('x', results.best_parameters)
        self.assertIn('y', results.best_parameters)
        self.assertIsInstance(results.best_value, float)
        self.assertIsInstance(results.quantum_advantage_score, float)
    
    def test_gaussian_process_prediction(self):
        """Test Gaussian process prediction"""
        # Add some observations
        self.optimizer.X_observed = [np.array([0.0, 0.0]), np.array([1.0, 1.0])]
        self.optimizer.y_observed = [0.0, 2.0]
        
        # Make prediction
        mean, variance = self.optimizer._predict_gp(np.array([0.5, 0.5]))
        
        self.assertIsInstance(mean, float)
        self.assertIsInstance(variance, float)
        self.assertGreater(variance, 0)
    
    def test_acquisition_functions(self):
        """Test quantum-enhanced acquisition functions"""
        # Setup some mock data
        self.optimizer.X_observed = [np.array([0.0]), np.array([1.0])]
        self.optimizer.y_observed = [1.0, 0.0]
        
        # Test expected improvement
        ei = self.optimizer._quantum_expected_improvement(np.array([0.5]))
        self.assertIsInstance(ei, float)
        self.assertGreater(ei, 0)
        
        # Test upper confidence bound
        ucb = self.optimizer._quantum_upper_confidence_bound(np.array([0.5]))
        self.assertIsInstance(ucb, float)

class TestDistributedQuantumOptimization(unittest.TestCase):
    """Test distributed quantum optimization framework"""
    
    def setUp(self):
        self.cluster_config = {
            'local_workers': 4,
            'remote_workers': []
        }
        self.optimizer = DistributedQuantumOptimizer(
            cluster_config=self.cluster_config,
            enable_auto_scaling=True,
            max_workers=10
        )
    
    def test_initialization(self):
        """Test proper initialization"""
        self.assertEqual(self.optimizer.max_workers, 10)
        self.assertTrue(self.optimizer.enable_auto_scaling)
        self.assertTrue(self.optimizer.enable_fault_tolerance)
    
    @patch('asyncio.run')
    def test_worker_discovery(self, mock_asyncio_run):
        """Test worker node discovery"""
        async def mock_discover():
            await self.optimizer._discover_workers()
            return len(self.optimizer.workers)
        
        # Run discovery
        asyncio_run = __import__('asyncio').run
        worker_count = asyncio_run(mock_discover())
        
        self.assertGreater(worker_count, 0)
        self.assertLessEqual(worker_count, self.cluster_config['local_workers'])
    
    def test_task_creation(self):
        """Test optimization task creation"""
        task = OptimizationTask(
            task_id="test_task_1",
            problem_data={'dimension': 5},
            parameters={'max_iterations': 100},
            priority=TaskPriority.HIGH,
            quantum_required=True
        )
        
        self.assertEqual(task.task_id, "test_task_1")
        self.assertEqual(task.priority, TaskPriority.HIGH)
        self.assertTrue(task.quantum_required)
        self.assertEqual(task.problem_data['dimension'], 5)
    
    def test_worker_scoring(self):
        """Test worker selection scoring"""
        from quantum_hyper_search.optimization.distributed_quantum_optimization import WorkerNode, WorkerStatus
        
        # Create mock workers
        quantum_worker = WorkerNode(
            worker_id="quantum_worker",
            hostname="localhost",
            port=8000,
            status=WorkerStatus.IDLE,
            quantum_backend_available=True,
            cpu_count=4,
            memory_gb=16.0
        )
        
        classical_worker = WorkerNode(
            worker_id="classical_worker",
            hostname="localhost",
            port=8001,
            status=WorkerStatus.IDLE,
            quantum_backend_available=False,
            cpu_count=2,
            memory_gb=8.0
        )
        
        # Create quantum task
        quantum_task = OptimizationTask(
            task_id="quantum_task",
            problem_data={},
            parameters={},
            quantum_required=True
        )
        
        # Calculate scores
        quantum_score = self.optimizer._calculate_worker_score(quantum_worker, quantum_task)
        classical_score = self.optimizer._calculate_worker_score(classical_worker, quantum_task)
        
        # Quantum worker should score higher for quantum tasks
        self.assertGreater(quantum_score, classical_score)

class TestAdaptiveResourceManagement(unittest.TestCase):
    """Test adaptive resource management system"""
    
    def setUp(self):
        self.resource_manager = AdaptiveResourceManager(
            allocation_strategy=AllocationStrategy.ADAPTIVE_LEARNING,
            enable_quantum_awareness=True
        )
    
    def test_initialization(self):
        """Test proper initialization"""
        self.assertEqual(
            self.resource_manager.allocation_strategy,
            AllocationStrategy.ADAPTIVE_LEARNING
        )
        self.assertTrue(self.resource_manager.enable_quantum_awareness)
        self.assertIsNotNone(self.resource_manager.system_resources)
    
    def test_resource_detection(self):
        """Test system resource detection"""
        resources = self.resource_manager._detect_system_resources()
        
        # Should have all resource types
        from quantum_hyper_search.optimization.adaptive_resource_management import ResourceType
        for resource_type in ResourceType:
            self.assertIn(resource_type, resources)
            self.assertGreater(resources[resource_type], 0)
    
    def test_resource_request_and_allocation(self):
        """Test resource request and allocation process"""
        request = ResourceRequest(
            request_id="test_request_1",
            task_id="test_task_1",
            priority=3,
            estimated_duration=60.0,
            cpu_cores=2,
            memory_gb=4.0,
            quantum_qpu_time=10.0
        )
        
        allocation_id = self.resource_manager.request_resources(request)
        
        if allocation_id:  # If resources were available
            self.assertIsInstance(allocation_id, str)
            self.assertIn(allocation_id, self.resource_manager.active_allocations)
            
            # Release resources
            success = self.resource_manager.release_resources(allocation_id)
            self.assertTrue(success)
            self.assertNotIn(allocation_id, self.resource_manager.active_allocations)
    
    def test_quantum_aware_allocation(self):
        """Test quantum-aware allocation strategy"""
        self.resource_manager.allocation_strategy = AllocationStrategy.QUANTUM_AWARE
        
        quantum_request = ResourceRequest(
            request_id="quantum_request",
            task_id="quantum_task",
            priority=3,
            estimated_duration=30.0,
            cpu_cores=1,
            memory_gb=2.0,
            quantum_qpu_time=20.0,
            quantum_advantage_expected=True
        )
        
        allocation = self.resource_manager._quantum_aware_allocation(quantum_request)
        
        from quantum_hyper_search.optimization.adaptive_resource_management import ResourceType
        
        # Should have enhanced quantum and memory allocation
        self.assertGreater(
            allocation[ResourceType.QUANTUM_QPU], 
            quantum_request.quantum_qpu_time
        )
        self.assertGreater(
            allocation[ResourceType.MEMORY], 
            quantum_request.memory_gb
        )
    
    def test_resource_monitoring(self):
        """Test resource monitoring capabilities"""
        # Start monitoring briefly
        self.resource_manager.start_monitoring()
        time.sleep(0.1)  # Let it collect one sample
        self.resource_manager.stop_monitoring()
        
        # Should have collected metrics
        self.assertIsNotNone(self.resource_manager.current_metrics)
        self.assertGreaterEqual(self.resource_manager.current_metrics.cpu_usage_percent, 0)

class TestIntegrationScenarios(unittest.TestCase):
    """Test integration scenarios combining multiple research components"""
    
    def setUp(self):
        self.backend = MockQuantumBackend()
    
    def test_parallel_tempering_with_error_correction(self):
        """Test integration of parallel tempering with error correction"""
        # Setup parallel tempering
        tempering_params = TemperingParams(temperatures=[0.5, 1.0], exchange_attempts=5)
        tempering_optimizer = QuantumParallelTempering(
            backend=self.backend,
            tempering_params=tempering_params
        )
        
        # Setup error correction
        correction_params = ErrorCorrectionParams(repetition_code_distance=3)
        error_corrector = QuantumErrorCorrection(
            backend=self.backend,
            correction_params=correction_params
        )
        
        # Simple QUBO problem
        qubo_matrix = np.array([[1, -0.5], [-0.5, 1]])
        
        # Run parallel tempering
        tempering_results = tempering_optimizer.optimize(
            qubo_matrix=qubo_matrix,
            max_iterations=10
        )
        
        # Apply error correction to the result
        correction_results = error_corrector.correct_qubo_solution(
            qubo_matrix=qubo_matrix,
            initial_solution=tempering_results.best_solution
        )
        
        # Both should complete successfully
        self.assertIsNotNone(tempering_results.best_solution)
        self.assertIsNotNone(correction_results.corrected_solution)
    
    def test_bayesian_optimization_with_quantum_walks(self):
        """Test integration of Bayesian optimization with quantum walks"""
        # Setup Bayesian optimizer
        bayes_optimizer = QuantumBayesianOptimizer(
            backend=self.backend,
            bayes_params=BayesianOptParams(max_evaluations=5)
        )
        
        # Setup quantum walk optimizer
        walk_optimizer = QuantumWalkOptimizer(
            backend=self.backend,
            walk_params=QuantumWalkParams(walk_length=10)
        )
        
        def hybrid_objective(params):
            """Objective function that uses both optimizers"""
            # Use quantum walk for discrete part
            discrete_result = walk_optimizer.optimize(
                objective_function=lambda x: sum(x),
                search_space_dim=2,
                max_iterations=5
            )
            
            # Combine with continuous parameters
            return params['x'] ** 2 + discrete_result.best_energy
        
        # Run Bayesian optimization
        results = bayes_optimizer.optimize(
            objective_function=hybrid_objective,
            parameter_bounds={'x': (-1.0, 1.0)},
            n_initial_points=2
        )
        
        self.assertIsNotNone(results.best_parameters)
        self.assertIn('x', results.best_parameters)

if __name__ == '__main__':
    # Configure logging for tests
    import logging
    logging.basicConfig(level=logging.INFO)
    
    # Run test suite
    unittest.main(verbosity=2)