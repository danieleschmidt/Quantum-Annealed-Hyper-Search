"""
Optimization history tracking and analysis.
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from dataclasses import dataclass
from datetime import datetime


@dataclass
class EvaluationRecord:
    """Single hyperparameter evaluation record."""
    parameters: Dict[str, Any]
    score: float
    iteration: int
    timestamp: datetime
    metadata: Optional[Dict[str, Any]] = None


class OptimizationHistory:
    """
    Tracks and analyzes optimization history for quantum hyperparameter search.
    """
    
    def __init__(self):
        """Initialize optimization history."""
        self.evaluations: List[EvaluationRecord] = []
        self._best_score = float('-inf')
        self._best_params = None
        
    def add_evaluation(
        self,
        parameters: Dict[str, Any],
        score: float,
        iteration: int,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Add a new evaluation to the history.
        
        Args:
            parameters: Parameter configuration that was evaluated
            score: Objective score achieved
            iteration: Optimization iteration number
            metadata: Additional metadata about the evaluation
        """
        record = EvaluationRecord(
            parameters=parameters.copy(),
            score=score,
            iteration=iteration,
            timestamp=datetime.now(),
            metadata=metadata or {}
        )
        
        self.evaluations.append(record)
        
        # Update best
        if score > self._best_score:
            self._best_score = score
            self._best_params = parameters.copy()
    
    @property
    def best_score(self) -> float:
        """Get the best score achieved so far."""
        return self._best_score
    
    @property
    def best_params(self) -> Optional[Dict[str, Any]]:
        """Get the best parameters found so far."""
        return self._best_params
    
    @property
    def n_evaluations(self) -> int:
        """Get total number of evaluations."""
        return len(self.evaluations)
    
    def get_convergence_data(self) -> Tuple[List[int], List[float]]:
        """
        Get convergence data for plotting.
        
        Returns:
            Tuple of (iteration_numbers, best_scores_so_far)
        """
        if not self.evaluations:
            return [], []
            
        iterations = []
        best_scores = []
        current_best = float('-inf')
        
        for eval_record in self.evaluations:
            iterations.append(eval_record.iteration)
            current_best = max(current_best, eval_record.score)
            best_scores.append(current_best)
        
        return iterations, best_scores
    
    def get_scores_by_iteration(self) -> Dict[int, List[float]]:
        """
        Group scores by iteration.
        
        Returns:
            Dictionary mapping iteration numbers to lists of scores
        """
        scores_by_iter = {}
        
        for eval_record in self.evaluations:
            iteration = eval_record.iteration
            if iteration not in scores_by_iter:
                scores_by_iter[iteration] = []
            scores_by_iter[iteration].append(eval_record.score)
        
        return scores_by_iter
    
    def get_parameter_importance(self) -> Dict[str, float]:
        """
        Estimate parameter importance using variance analysis.
        
        Returns:
            Dictionary mapping parameter names to importance scores
        """
        if len(self.evaluations) < 10:
            return {}
        
        # Convert to DataFrame for easier analysis
        df_data = []
        for eval_record in self.evaluations:
            row = eval_record.parameters.copy()
            row['score'] = eval_record.score
            df_data.append(row)
        
        df = pd.DataFrame(df_data)
        
        importance = {}
        
        for param_name in df.columns:
            if param_name == 'score':
                continue
                
            try:
                # Calculate correlation with score for numerical parameters
                if df[param_name].dtype in ['int64', 'float64']:
                    correlation = abs(df[param_name].corr(df['score']))
                    importance[param_name] = correlation if not np.isnan(correlation) else 0.0
                else:
                    # For categorical parameters, use ANOVA-like measure
                    grouped = df.groupby(param_name)['score']
                    between_var = grouped.mean().var()
                    within_var = grouped.apply(lambda x: x.var()).mean()
                    
                    if within_var > 0:
                        f_score = between_var / within_var
                        importance[param_name] = min(f_score / 10.0, 1.0)  # Normalize
                    else:
                        importance[param_name] = 0.0
                        
            except Exception:
                importance[param_name] = 0.0
        
        return importance
    
    def get_top_configurations(self, n: int = 10) -> List[EvaluationRecord]:
        """
        Get top N configurations by score.
        
        Args:
            n: Number of top configurations to return
            
        Returns:
            List of top evaluation records
        """
        sorted_evals = sorted(self.evaluations, key=lambda x: x.score, reverse=True)
        return sorted_evals[:n]
    
    def to_dataframe(self) -> pd.DataFrame:
        """
        Convert optimization history to pandas DataFrame.
        
        Returns:
            DataFrame with all evaluation records
        """
        if not self.evaluations:
            return pd.DataFrame()
        
        data = []
        for eval_record in self.evaluations:
            row = eval_record.parameters.copy()
            row.update({
                'score': eval_record.score,
                'iteration': eval_record.iteration,
                'timestamp': eval_record.timestamp
            })
            if eval_record.metadata:
                row.update(eval_record.metadata)
            data.append(row)
        
        return pd.DataFrame(data)
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get summary statistics for the optimization run.
        
        Returns:
            Dictionary with summary statistics
        """
        if not self.evaluations:
            return {}
        
        scores = [eval_record.score for eval_record in self.evaluations]
        
        return {
            'n_evaluations': len(self.evaluations),
            'best_score': self.best_score,
            'worst_score': min(scores),
            'mean_score': np.mean(scores),
            'std_score': np.std(scores),
            'score_range': max(scores) - min(scores),
            'n_iterations': len(set(eval_record.iteration for eval_record in self.evaluations)),
            'evaluations_per_iteration': len(self.evaluations) / len(set(eval_record.iteration for eval_record in self.evaluations)),
        }