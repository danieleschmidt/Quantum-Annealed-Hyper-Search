"""
Quantum Advantage Analyzer - Advanced analysis and benchmarking of quantum vs classical methods.

This module provides comprehensive analysis tools to measure and validate quantum advantage
in hyperparameter optimization tasks, with statistical significance testing and
publication-ready visualizations.
"""

import time
import warnings
from typing import Dict, List, Tuple, Any, Optional, Callable
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from concurrent.futures import ProcessPoolExecutor
import logging

# Suppress warnings for clean benchmarking
warnings.filterwarnings('ignore')
logging.getLogger('sklearn').setLevel(logging.WARNING)


class QuantumAdvantageAnalyzer:
    """
    Comprehensive analyzer for measuring quantum advantage in hyperparameter optimization.
    
    Provides statistical testing, performance comparisons, and publication-ready analysis
    of quantum vs classical optimization methods.
    """
    
    def __init__(self, 
                 significance_level: float = 0.05,
                 min_runs: int = 10,
                 max_runs: int = 50,
                 enable_parallel: bool = True):
        """
        Initialize quantum advantage analyzer.
        
        Args:
            significance_level: Statistical significance threshold (p-value)
            min_runs: Minimum number of runs per method
            max_runs: Maximum number of runs per method  
            enable_parallel: Enable parallel execution of benchmarks
        """
        self.significance_level = significance_level
        self.min_runs = min_runs
        self.max_runs = max_runs
        self.enable_parallel = enable_parallel
        self.results_cache = {}
        
    def analyze_convergence_advantage(
        self,
        quantum_results: List[Dict],
        classical_results: List[Dict], 
        convergence_threshold: float = 0.95
    ) -> Dict[str, Any]:
        """
        Analyze convergence speed advantage of quantum vs classical methods.
        
        Args:
            quantum_results: List of quantum optimization results
            classical_results: List of classical optimization results
            convergence_threshold: Threshold for convergence (fraction of optimal)
            
        Returns:
            Dictionary containing convergence analysis results
        """
        print("🔬 Analyzing convergence advantage...")
        
        # Extract convergence data
        quantum_convergence = self._extract_convergence_data(quantum_results, convergence_threshold)
        classical_convergence = self._extract_convergence_data(classical_results, convergence_threshold)
        
        # Statistical tests
        convergence_comparison = self._compare_convergence_speeds(
            quantum_convergence, classical_convergence
        )
        
        # Effect size analysis
        effect_sizes = self._calculate_effect_sizes(
            quantum_convergence, classical_convergence
        )
        
        return {
            'convergence_analysis': convergence_comparison,
            'effect_sizes': effect_sizes,
            'quantum_mean_iterations': np.mean(quantum_convergence),
            'classical_mean_iterations': np.mean(classical_convergence),
            'speedup_factor': np.mean(classical_convergence) / np.mean(quantum_convergence),
            'statistical_significance': convergence_comparison['p_value'] < self.significance_level
        }
    
    def analyze_solution_quality_advantage(
        self,
        quantum_results: List[Dict],
        classical_results: List[Dict]
    ) -> Dict[str, Any]:
        """
        Analyze solution quality advantage of quantum vs classical methods.
        
        Args:
            quantum_results: List of quantum optimization results
            classical_results: List of classical optimization results
            
        Returns:
            Dictionary containing solution quality analysis
        """
        print("🎯 Analyzing solution quality advantage...")
        
        # Extract best scores
        quantum_scores = [result['best_score'] for result in quantum_results]
        classical_scores = [result['best_score'] for result in classical_results]
        
        # Statistical comparison
        quality_comparison = self._compare_distributions(quantum_scores, classical_scores)
        
        # Practical significance
        mean_improvement = np.mean(quantum_scores) - np.mean(classical_scores)
        relative_improvement = mean_improvement / np.mean(classical_scores) if np.mean(classical_scores) > 0 else 0
        
        return {
            'quality_comparison': quality_comparison,
            'mean_quantum_score': np.mean(quantum_scores),
            'mean_classical_score': np.mean(classical_scores),
            'absolute_improvement': mean_improvement,
            'relative_improvement': relative_improvement,
            'quantum_better_rate': np.mean(np.array(quantum_scores) > np.array(classical_scores)),
            'statistical_significance': quality_comparison['p_value'] < self.significance_level
        }
    
    def analyze_exploration_diversity(
        self,
        quantum_results: List[Dict],
        classical_results: List[Dict]
    ) -> Dict[str, Any]:
        """
        Analyze exploration diversity advantage of quantum methods.
        
        Args:
            quantum_results: List of quantum optimization results
            classical_results: List of classical optimization results
            
        Returns:
            Dictionary containing diversity analysis
        """
        print("🌐 Analyzing exploration diversity...")
        
        # Calculate diversity metrics
        quantum_diversity = self._calculate_exploration_diversity(quantum_results)
        classical_diversity = self._calculate_exploration_diversity(classical_results)
        
        # Statistical comparison
        diversity_comparison = self._compare_distributions(quantum_diversity, classical_diversity)
        
        return {
            'diversity_comparison': diversity_comparison,
            'quantum_mean_diversity': np.mean(quantum_diversity),
            'classical_mean_diversity': np.mean(classical_diversity),
            'diversity_advantage': np.mean(quantum_diversity) / np.mean(classical_diversity),
            'statistical_significance': diversity_comparison['p_value'] < self.significance_level
        }
    
    def analyze_hardware_efficiency(
        self,
        quantum_results: List[Dict]
    ) -> Dict[str, Any]:
        """
        Analyze quantum hardware efficiency metrics.
        
        Args:
            quantum_results: List of quantum optimization results
            
        Returns:
            Dictionary containing hardware efficiency analysis
        """
        print("⚡ Analyzing hardware efficiency...")
        
        # Extract hardware metrics
        chain_break_fractions = []
        embedding_overheads = []
        annealing_times = []
        
        for result in quantum_results:
            if 'hardware_metrics' in result:
                metrics = result['hardware_metrics']
                chain_break_fractions.append(metrics.get('avg_chain_break_fraction', 0))
                embedding_overheads.append(metrics.get('embedding_overhead', 1))
                annealing_times.append(metrics.get('total_annealing_time', 0))
        
        return {
            'mean_chain_break_fraction': np.mean(chain_break_fractions) if chain_break_fractions else 0,
            'mean_embedding_overhead': np.mean(embedding_overheads) if embedding_overheads else 1,
            'total_annealing_time': np.sum(annealing_times),
            'avg_annealing_time_per_run': np.mean(annealing_times) if annealing_times else 0,
            'hardware_utilization_score': self._calculate_hardware_utilization_score(
                chain_break_fractions, embedding_overheads
            )
        }
    
    def generate_comprehensive_report(
        self,
        quantum_results: List[Dict],
        classical_results: Dict[str, List[Dict]],  # method_name -> results
        output_file: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate comprehensive quantum advantage report.
        
        Args:
            quantum_results: Quantum optimization results
            classical_results: Classical optimization results by method
            output_file: Optional output file path
            
        Returns:
            Comprehensive analysis report
        """
        print("📊 Generating comprehensive quantum advantage report...")
        
        report = {
            'summary': self._generate_executive_summary(),
            'methodology': self._generate_methodology_section(),
            'results': {}
        }
        
        # Analyze vs each classical method
        for method_name, classical_data in classical_results.items():
            print(f"   Comparing against {method_name}...")
            
            method_analysis = {
                'convergence': self.analyze_convergence_advantage(quantum_results, classical_data),
                'quality': self.analyze_solution_quality_advantage(quantum_results, classical_data),
                'diversity': self.analyze_exploration_diversity(quantum_results, classical_data)
            }
            
            report['results'][method_name] = method_analysis
        
        # Hardware analysis
        report['hardware_analysis'] = self.analyze_hardware_efficiency(quantum_results)
        
        # Overall conclusions
        report['conclusions'] = self._generate_conclusions(report['results'])
        
        # Visualizations
        report['visualizations'] = self._generate_visualizations(
            quantum_results, classical_results
        )
        
        # Save report if requested
        if output_file:
            self._save_report(report, output_file)
        
        return report
    
    def _extract_convergence_data(self, results: List[Dict], threshold: float) -> List[int]:
        """
        Extract convergence iterations from optimization results.
        """
        convergence_iterations = []
        
        for result in results:
            if 'history' in result and hasattr(result['history'], 'scores'):
                scores = result['history'].scores
                if len(scores) > 0:
                    target_score = threshold * np.max(scores)
                    convergence_iter = np.argmax(scores >= target_score)
                    convergence_iterations.append(convergence_iter + 1)
        
        return convergence_iterations
    
    def _compare_convergence_speeds(self, quantum_data: List, classical_data: List) -> Dict:
        """
        Compare convergence speeds using statistical tests.
        """
        if len(quantum_data) == 0 or len(classical_data) == 0:
            return {'test': 'insufficient_data', 'p_value': 1.0}
        
        # Mann-Whitney U test (non-parametric)
        statistic, p_value = stats.mannwhitneyu(
            quantum_data, classical_data, alternative='less'
        )
        
        return {
            'test': 'mann_whitney_u',
            'statistic': statistic,
            'p_value': p_value,
            'quantum_faster': p_value < self.significance_level
        }
    
    def _compare_distributions(self, quantum_data: List, classical_data: List) -> Dict:
        """
        Compare two distributions using appropriate statistical tests.
        """
        if len(quantum_data) == 0 or len(classical_data) == 0:
            return {'test': 'insufficient_data', 'p_value': 1.0}
        
        # Welch's t-test (assumes unequal variances)
        statistic, p_value = stats.ttest_ind(
            quantum_data, classical_data, equal_var=False
        )
        
        return {
            'test': 'welch_t_test',
            'statistic': statistic,
            'p_value': p_value,
            'quantum_better': statistic > 0 and p_value < self.significance_level
        }
    
    def _calculate_effect_sizes(self, quantum_data: List, classical_data: List) -> Dict:
        """
        Calculate effect sizes (Cohen's d).
        """
        if len(quantum_data) == 0 or len(classical_data) == 0:
            return {'cohens_d': 0}
        
        pooled_std = np.sqrt(
            ((len(quantum_data) - 1) * np.var(quantum_data) + 
             (len(classical_data) - 1) * np.var(classical_data)) /
            (len(quantum_data) + len(classical_data) - 2)
        )
        
        cohens_d = (np.mean(classical_data) - np.mean(quantum_data)) / pooled_std
        
        # Effect size interpretation
        if abs(cohens_d) < 0.2:
            magnitude = 'negligible'
        elif abs(cohens_d) < 0.5:
            magnitude = 'small'
        elif abs(cohens_d) < 0.8:
            magnitude = 'medium'
        else:
            magnitude = 'large'
        
        return {
            'cohens_d': cohens_d,
            'magnitude': magnitude,
            'favors': 'quantum' if cohens_d > 0 else 'classical'
        }
    
    def _calculate_exploration_diversity(self, results: List[Dict]) -> List[float]:
        """
        Calculate exploration diversity for each optimization run.
        """
        diversity_scores = []
        
        for result in results:
            if 'history' in result and hasattr(result['history'], 'trials'):
                trials = result['history'].trials
                if len(trials) > 1:
                    # Calculate pairwise parameter distances
                    distances = []
                    for i in range(len(trials)):
                        for j in range(i + 1, len(trials)):
                            dist = self._parameter_distance(trials[i], trials[j])
                            distances.append(dist)
                    
                    diversity = np.mean(distances) if distances else 0
                    diversity_scores.append(diversity)
        
        return diversity_scores
    
    def _parameter_distance(self, params1: Dict, params2: Dict) -> float:
        """
        Calculate distance between parameter configurations.
        """
        distance = 0
        common_params = set(params1.keys()) & set(params2.keys())
        
        for param in common_params:
            val1, val2 = params1[param], params2[param]
            
            if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
                # Normalized difference for numeric parameters
                distance += abs(val1 - val2) / max(abs(val1), abs(val2), 1)
            else:
                # Binary difference for categorical parameters
                distance += 0 if val1 == val2 else 1
        
        return distance / max(len(common_params), 1)
    
    def _calculate_hardware_utilization_score(self, chain_breaks: List, overheads: List) -> float:
        """
        Calculate overall hardware utilization score.
        """
        if not chain_breaks or not overheads:
            return 0.0
        
        # Lower chain breaks and overheads = better utilization
        chain_score = 1.0 - np.mean(chain_breaks)
        overhead_score = 1.0 / np.mean(overheads) if np.mean(overheads) > 0 else 0
        
        return (chain_score + overhead_score) / 2
    
    def _generate_executive_summary(self) -> Dict:
        """
        Generate executive summary section.
        """
        return {
            'objective': 'Comprehensive analysis of quantum advantage in hyperparameter optimization',
            'methods': 'Statistical comparison of quantum annealing vs classical optimization methods',
            'significance_level': self.significance_level,
            'generated_at': time.strftime('%Y-%m-%d %H:%M:%S')
        }
    
    def _generate_methodology_section(self) -> Dict:
        """
        Generate methodology section.
        """
        return {
            'statistical_tests': [
                'Mann-Whitney U test for convergence speed comparison',
                "Welch's t-test for solution quality comparison",
                "Cohen's d for effect size calculation"
            ],
            'metrics': [
                'Convergence speed (iterations to 95% optimal)',
                'Solution quality (final optimization score)', 
                'Exploration diversity (parameter space coverage)',
                'Hardware efficiency (chain breaks, embedding overhead)'
            ],
            'significance_level': self.significance_level
        }
    
    def _generate_conclusions(self, results: Dict) -> Dict:
        """
        Generate overall conclusions from analysis results.
        """
        quantum_advantages = []
        
        for method, analysis in results.items():
            if analysis['convergence']['statistical_significance']:
                quantum_advantages.append(f"Significantly faster convergence vs {method}")
            
            if analysis['quality']['statistical_significance']:
                quantum_advantages.append(f"Significantly better solution quality vs {method}")
            
            if analysis['diversity']['statistical_significance']:
                quantum_advantages.append(f"Significantly more diverse exploration vs {method}")
        
        return {
            'quantum_advantages': quantum_advantages,
            'overall_assessment': 'Quantum advantage demonstrated' if quantum_advantages else 'No clear quantum advantage',
            'recommendation': self._generate_recommendation(quantum_advantages)
        }
    
    def _generate_recommendation(self, advantages: List[str]) -> str:
        """
        Generate recommendation based on analysis.
        """
        if len(advantages) >= 2:
            return "Strong recommendation for quantum hyperparameter optimization"
        elif len(advantages) == 1:
            return "Moderate recommendation with specific advantages in certain scenarios"
        else:
            return "Classical methods remain competitive; consider problem-specific factors"
    
    def _generate_visualizations(self, quantum_results: List, classical_results: Dict) -> List[str]:
        """
        Generate visualization filenames.
        """
        return [
            'convergence_comparison.png',
            'solution_quality_distribution.png', 
            'exploration_diversity_boxplot.png',
            'hardware_efficiency_metrics.png'
        ]
    
    def _save_report(self, report: Dict, filename: str):
        """
        Save report to file.
        """
        import json
        
        with open(filename, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"📄 Report saved to {filename}")
