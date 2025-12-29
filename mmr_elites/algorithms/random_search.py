"""
Random Search Baseline.
"""

import numpy as np
import time
from typing import Dict, Tuple, Optional
from .base import QDAlgorithm, QDResult
from ..metrics.qd_metrics import compute_all_metrics


class RandomSearch(QDAlgorithm):
    """
    Random Search baseline.
    Generates random solutions and keeps the best K (by fitness) or all of them.
    For fair comparison, we usually treat it as a "filter" that keeps top-K.
    """
    
    def __init__(self, archive_size: int):
        self.archive_size = archive_size
        self.archive = np.array([])
        self.fitness = np.array([])
        self.descriptors = np.array([])
    
    def run(
        self,
        task,
        generations: int,
        batch_size: int,
        mutation_sigma: float, # Unused
        seed: int,
        log_interval: int = 100,
    ) -> QDResult:
        np.random.seed(seed)
        
        if hasattr(task, 'genome_dim'):
            n_dof = task.genome_dim
        elif hasattr(task, 'n_dof'):
            n_dof = task.n_dof
        else:
            n_dof = 20
            
        history = {
            "generation": [],
            "qd_score": [],
            "qd_score_at_budget": [],
            "max_fitness": [],
            "mean_fitness": [],
            "mean_pairwise_distance": [],
            "uniformity_cv": [],
            "archive_size": [],
        }
        
        start_time = time.time()
        
        # We simulate "generations" to match other algos
        # Total evaluations = generations * batch_size
        # We keep ALL solutions found, but metrics will look at top-K
        
        all_genomes = []
        all_fitness = []
        all_descriptors = []
        
        for gen in range(1, generations + 1):
            # Generate random batch
            batch = np.random.uniform(-np.pi, np.pi, (batch_size, n_dof))
            fit, desc = task.evaluate(batch)
            
            all_genomes.append(batch)
            all_fitness.append(fit)
            all_descriptors.append(desc)
            
            if gen % log_interval == 0:
                # Concatenate current results
                curr_fit = np.concatenate(all_fitness)
                curr_desc = np.vstack(all_descriptors)
                
                # Keep top K for metrics (simulating an archive)
                if len(curr_fit) > self.archive_size:
                    # Sort by fitness
                    top_k_idx = np.argsort(curr_fit)[-self.archive_size:]
                    metric_fit = curr_fit[top_k_idx]
                    metric_desc = curr_desc[top_k_idx]
                else:
                    metric_fit = curr_fit
                    metric_desc = curr_desc
                
                metrics = compute_all_metrics(
                    metric_fit, metric_desc, budget_k=self.archive_size
                )
                
                history["generation"].append(gen)
                for key in history:
                    if key != "generation" and key in metrics:
                        history[key].append(metrics[key])
        
        runtime = time.time() - start_time
        
        # Final aggregation
        self.archive = np.vstack(all_genomes)
        self.fitness = np.concatenate(all_fitness)
        self.descriptors = np.vstack(all_descriptors)
        
        # Select top K for final result
        if len(self.fitness) > self.archive_size:
            top_k_idx = np.argsort(self.fitness)[-self.archive_size:]
            final_genomes = self.archive[top_k_idx]
            final_fitness = self.fitness[top_k_idx]
            final_descriptors = self.descriptors[top_k_idx]
        else:
            final_genomes = self.archive
            final_fitness = self.fitness
            final_descriptors = self.descriptors

        return QDResult(
            algorithm="RandomSearch",
            seed=seed,
            runtime=runtime,
            final_metrics=compute_all_metrics(
                final_fitness, final_descriptors, budget_k=self.archive_size
            ),
            history=history,
            final_genomes=final_genomes,
            final_fitness=final_fitness,
            final_descriptors=final_descriptors,
        )

    def get_archive_size(self) -> int:
        return self.archive_size
