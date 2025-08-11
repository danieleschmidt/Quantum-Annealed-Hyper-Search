"""
Quantum Error Correction for QUBO Optimization

Implementation of error correction techniques specifically designed for
quantum annealing-based QUBO optimization to improve solution quality.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
import logging
from dataclasses import dataclass
from collections import Counter
import time
from ..core.base import QuantumBackend
from ..utils.logging import setup_logger

logger = setup_logger(__name__)

@dataclass
class ErrorCorrectionParams:
    """Parameters for quantum error correction"""
    repetition_code_distance: int = 3
    majority_voting_threshold: float = 0.6
    error_detection_rounds: int = 5
    correction_strength: float = 0.1
    adaptive_correction: bool = True

@dataclass
class CorrectionResults:
    """Results from quantum error correction"""
    original_solution: Dict[str, Any]
    corrected_solution: Dict[str, Any]
    error_rate: float
    corrections_applied: int
    confidence_score: float
    improvement_achieved: bool

class QuantumErrorCorrection:
    """
    Advanced quantum error correction system for QUBO optimization
    that uses repetition codes and majority voting to improve solution quality.
    """
    
    def __init__(
        self,
        backend: QuantumBackend,
        correction_params: ErrorCorrectionParams,
        enable_adaptive_correction: bool = True
    ):
        self.backend = backend
        self.params = correction_params
        self.enable_adaptive_correction = enable_adaptive_correction
        self.error_history = []
        self.correction_statistics = {
            'total_corrections': 0,
            'successful_corrections': 0,
            'error_rate_history': []
        }
    
    def correct_qubo_solution(
        self,
        qubo_matrix: np.ndarray,
        initial_solution: Optional[Dict[str, Any]] = None,
        num_correction_rounds: int = None
    ) -> CorrectionResults:
        """
        Apply quantum error correction to QUBO optimization
        
        Args:
            qubo_matrix: QUBO problem formulation
            initial_solution: Initial solution to correct (if None, generates new)
            num_correction_rounds: Number of correction rounds to apply
            
        Returns:
            CorrectionResults with corrected solution and statistics
        """
        
        if num_correction_rounds is None:
            num_correction_rounds = self.params.error_detection_rounds
        
        # Get initial solution
        if initial_solution is None:
            logger.info("Generating initial solution for error correction")
            original_solution = self._generate_initial_solution(qubo_matrix)
        else:
            original_solution = initial_solution
        
        logger.info(f"Starting error correction with {num_correction_rounds} rounds")
        
        # Apply repetition coding
        encoded_solutions = self._apply_repetition_code(qubo_matrix, original_solution)
        
        # Detect and correct errors
        corrected_solution, corrections_applied = self._detect_and_correct_errors(
            encoded_solutions, qubo_matrix
        )
        
        # Calculate correction metrics
        error_rate = self._calculate_error_rate(original_solution, encoded_solutions)
        confidence_score = self._calculate_confidence_score(encoded_solutions)
        improvement = self._assess_improvement(original_solution, corrected_solution, qubo_matrix)
        
        # Update statistics
        self._update_correction_statistics(error_rate, corrections_applied, improvement)
        
        return CorrectionResults(
            original_solution=original_solution,
            corrected_solution=corrected_solution,
            error_rate=error_rate,
            corrections_applied=corrections_applied,
            confidence_score=confidence_score,
            improvement_achieved=improvement
        )
    
    def _generate_initial_solution(self, qubo_matrix: np.ndarray) -> Dict[str, Any]:
        """Generate initial solution using quantum backend"""
        try:
            result = self.backend.sample_qubo(
                Q=qubo_matrix,
                num_reads=100,
                temperature=1.0
            )
            
            best_sample = min(result.data(['sample', 'energy']), key=lambda x: x.energy)
            
            solution = {
                'variables': {str(i): best_sample.sample.get(i, 0) 
                             for i in range(qubo_matrix.shape[0])},
                'energy': best_sample.energy,
                'num_occurrences': getattr(best_sample, 'num_occurrences', 1)
            }
            
            return solution
            
        except Exception as e:
            logger.warning(f"Quantum backend failed, using classical fallback: {e}")
            return self._generate_classical_solution(qubo_matrix)
    
    def _generate_classical_solution(self, qubo_matrix: np.ndarray) -> Dict[str, Any]:
        """Generate classical solution as fallback"""
        n_vars = qubo_matrix.shape[0]
        
        # Simple greedy approach
        state = np.zeros(n_vars)
        current_energy = float(state.T @ qubo_matrix @ state)
        
        for i in range(n_vars):
            # Try flipping bit i
            test_state = state.copy()
            test_state[i] = 1
            test_energy = float(test_state.T @ qubo_matrix @ test_state)
            
            if test_energy < current_energy:
                state[i] = 1
                current_energy = test_energy
        
        return {
            'variables': {str(i): int(state[i]) for i in range(n_vars)},
            'energy': current_energy,
            'num_occurrences': 1
        }
    
    def _apply_repetition_code(
        self, 
        qubo_matrix: np.ndarray, 
        solution: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Apply repetition code by generating multiple solution instances"""
        
        encoded_solutions = []
        distance = self.params.repetition_code_distance
        
        logger.info(f"Applying repetition code with distance {distance}")
        
        for rep in range(distance):
            try:
                # Generate variation of the solution with small perturbations
                perturbed_solution = self._create_solution_variation(
                    solution, qubo_matrix, perturbation_strength=0.1 * rep
                )
                encoded_solutions.append(perturbed_solution)
                
            except Exception as e:
                logger.warning(f"Failed to create solution variation {rep}: {e}")
                # Use original solution as fallback
                encoded_solutions.append(solution.copy())
        
        return encoded_solutions
    
    def _create_solution_variation(
        self, 
        base_solution: Dict[str, Any], 
        qubo_matrix: np.ndarray,
        perturbation_strength: float
    ) -> Dict[str, Any]:
        """Create a variation of the base solution with controlled perturbation"""
        
        variables = base_solution['variables']
        n_vars = len(variables)
        
        # Create state vector
        state = np.array([variables[str(i)] for i in range(n_vars)])
        
        # Apply controlled perturbation
        if perturbation_strength > 0:
            n_flips = max(1, int(perturbation_strength * n_vars))
            flip_indices = np.random.choice(n_vars, size=n_flips, replace=False)
            
            for idx in flip_indices:
                state[idx] = 1 - state[idx]
        
        # Re-optimize locally
        optimized_state = self._local_optimization(state, qubo_matrix)
        
        # Calculate energy
        energy = float(optimized_state.T @ qubo_matrix @ optimized_state)
        
        return {
            'variables': {str(i): int(optimized_state[i]) for i in range(n_vars)},
            'energy': energy,
            'num_occurrences': 1,
            'perturbation_applied': perturbation_strength
        }
    
    def _local_optimization(self, state: np.ndarray, qubo_matrix: np.ndarray) -> np.ndarray:
        """Perform local optimization to improve the perturbed state"""
        
        current_state = state.copy()
        current_energy = float(current_state.T @ qubo_matrix @ current_state)
        
        # Simple hill climbing
        max_iterations = min(50, len(state))  # Limit iterations
        
        for _ in range(max_iterations):
            improved = False
            
            for i in range(len(state)):
                # Try flipping bit i
                test_state = current_state.copy()
                test_state[i] = 1 - test_state[i]
                test_energy = float(test_state.T @ qubo_matrix @ test_state)
                
                if test_energy < current_energy:
                    current_state = test_state
                    current_energy = test_energy
                    improved = True
                    break
            
            if not improved:
                break
        
        return current_state
    
    def _detect_and_correct_errors(
        self, 
        encoded_solutions: List[Dict[str, Any]], 
        qubo_matrix: np.ndarray
    ) -> Tuple[Dict[str, Any], int]:
        """Detect and correct errors using majority voting"""
        
        if not encoded_solutions:
            raise ValueError("No encoded solutions provided for error correction")
        
        n_vars = len(encoded_solutions[0]['variables'])
        corrections_applied = 0
        
        # Collect votes for each variable
        variable_votes = {}
        for var_idx in range(n_vars):
            var_key = str(var_idx)
            votes = [sol['variables'][var_key] for sol in encoded_solutions]
            variable_votes[var_key] = votes
        
        # Apply majority voting with confidence thresholding
        corrected_variables = {}
        
        for var_key, votes in variable_votes.items():
            vote_counts = Counter(votes)
            total_votes = len(votes)
            
            # Find majority value
            majority_value = vote_counts.most_common(1)[0][0]
            majority_count = vote_counts[majority_value]
            confidence = majority_count / total_votes
            
            if confidence >= self.params.majority_voting_threshold:
                corrected_variables[var_key] = majority_value
            else:
                # Use energy-based tie-breaking
                corrected_variables[var_key] = self._energy_based_tie_breaking(
                    var_key, votes, encoded_solutions, qubo_matrix
                )
                corrections_applied += 1
        
        # Calculate final energy
        corrected_state = np.array([corrected_variables[str(i)] for i in range(n_vars)])
        corrected_energy = float(corrected_state.T @ qubo_matrix @ corrected_state)
        
        corrected_solution = {
            'variables': corrected_variables,
            'energy': corrected_energy,
            'num_occurrences': len(encoded_solutions),
            'correction_method': 'majority_voting'
        }
        
        logger.info(f"Error correction completed: {corrections_applied} corrections applied")
        
        return corrected_solution, corrections_applied
    
    def _energy_based_tie_breaking(
        self,
        var_key: str,
        votes: List[int],
        encoded_solutions: List[Dict[str, Any]],
        qubo_matrix: np.ndarray
    ) -> int:
        """Use energy-based tie-breaking for ambiguous variables"""
        
        # Test both possible values and choose the one leading to lower energy
        var_idx = int(var_key)
        
        # Get base state (majority votes for all other variables)
        base_state = np.array([
            max(set([sol['variables'][str(i)] for sol in encoded_solutions]), 
                key=[sol['variables'][str(i)] for sol in encoded_solutions].count)
            for i in range(qubo_matrix.shape[0])
        ])
        
        # Test both values for this variable
        energies = {}
        for test_value in [0, 1]:
            test_state = base_state.copy()
            test_state[var_idx] = test_value
            energy = float(test_state.T @ qubo_matrix @ test_state)
            energies[test_value] = energy
        
        # Return value that gives lower energy
        return min(energies.keys(), key=lambda k: energies[k])
    
    def _calculate_error_rate(
        self, 
        original_solution: Dict[str, Any], 
        encoded_solutions: List[Dict[str, Any]]
    ) -> float:
        """Calculate the error rate across encoded solutions"""
        
        if len(encoded_solutions) <= 1:
            return 0.0
        
        original_vars = original_solution['variables']
        n_vars = len(original_vars)
        
        total_disagreements = 0
        total_comparisons = 0
        
        for i, sol1 in enumerate(encoded_solutions):
            for j, sol2 in enumerate(encoded_solutions[i+1:], i+1):
                for var_key in original_vars.keys():
                    if sol1['variables'][var_key] != sol2['variables'][var_key]:
                        total_disagreements += 1
                    total_comparisons += 1
        
        error_rate = total_disagreements / max(1, total_comparisons)
        self.error_history.append(error_rate)
        
        return error_rate
    
    def _calculate_confidence_score(self, encoded_solutions: List[Dict[str, Any]]) -> float:
        """Calculate confidence score based on solution consensus"""
        
        if len(encoded_solutions) <= 1:
            return 1.0
        
        n_vars = len(encoded_solutions[0]['variables'])
        consensus_scores = []
        
        for var_idx in range(n_vars):
            var_key = str(var_idx)
            votes = [sol['variables'][var_key] for sol in encoded_solutions]
            vote_counts = Counter(votes)
            majority_count = vote_counts.most_common(1)[0][1]
            consensus = majority_count / len(votes)
            consensus_scores.append(consensus)
        
        # Average consensus across all variables
        confidence = np.mean(consensus_scores)
        
        return float(confidence)
    
    def _assess_improvement(
        self,
        original_solution: Dict[str, Any],
        corrected_solution: Dict[str, Any],
        qubo_matrix: np.ndarray
    ) -> bool:
        """Assess whether error correction improved the solution"""
        
        original_energy = original_solution['energy']
        corrected_energy = corrected_solution['energy']
        
        improvement = corrected_energy < original_energy
        improvement_amount = original_energy - corrected_energy
        
        logger.info(f"Energy comparison: Original = {original_energy:.6f}, "
                   f"Corrected = {corrected_energy:.6f}, "
                   f"Improvement = {improvement_amount:.6f}")
        
        return improvement
    
    def _update_correction_statistics(
        self, 
        error_rate: float, 
        corrections_applied: int, 
        improvement: bool
    ):
        """Update internal correction statistics"""
        
        self.correction_statistics['total_corrections'] += corrections_applied
        if improvement:
            self.correction_statistics['successful_corrections'] += corrections_applied
        
        self.correction_statistics['error_rate_history'].append(error_rate)
        
        # Adaptive parameter adjustment
        if self.enable_adaptive_correction:
            self._adapt_correction_parameters(error_rate, improvement)
    
    def _adapt_correction_parameters(self, error_rate: float, improvement: bool):
        """Adaptively adjust correction parameters based on performance"""
        
        # Adjust repetition code distance based on error rate
        if error_rate > 0.3 and self.params.repetition_code_distance < 7:
            self.params.repetition_code_distance += 1
            logger.info(f"Increased repetition code distance to {self.params.repetition_code_distance}")
        elif error_rate < 0.1 and self.params.repetition_code_distance > 3:
            self.params.repetition_code_distance -= 1
            logger.info(f"Decreased repetition code distance to {self.params.repetition_code_distance}")
        
        # Adjust majority voting threshold based on improvement
        if not improvement and self.params.majority_voting_threshold > 0.5:
            self.params.majority_voting_threshold -= 0.05
            logger.info(f"Decreased majority voting threshold to {self.params.majority_voting_threshold:.2f}")
        elif improvement and self.params.majority_voting_threshold < 0.8:
            self.params.majority_voting_threshold += 0.02
            logger.info(f"Increased majority voting threshold to {self.params.majority_voting_threshold:.2f}")
    
    def get_correction_statistics(self) -> Dict[str, Any]:
        """Get comprehensive correction statistics"""
        
        success_rate = 0.0
        if self.correction_statistics['total_corrections'] > 0:
            success_rate = (self.correction_statistics['successful_corrections'] / 
                          self.correction_statistics['total_corrections'])
        
        avg_error_rate = 0.0
        if self.correction_statistics['error_rate_history']:
            avg_error_rate = np.mean(self.correction_statistics['error_rate_history'])
        
        return {
            'total_corrections': self.correction_statistics['total_corrections'],
            'successful_corrections': self.correction_statistics['successful_corrections'],
            'success_rate': success_rate,
            'average_error_rate': avg_error_rate,
            'current_repetition_distance': self.params.repetition_code_distance,
            'current_voting_threshold': self.params.majority_voting_threshold,
            'error_rate_history': self.correction_statistics['error_rate_history'][-10:]  # Last 10
        }