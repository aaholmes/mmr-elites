import numpy as np
from typing import Tuple, List

class Task:
    def evaluate(self, genomes: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Input: Batch of genomes (N, gene_dim)
        Output: 
            - fitnesses (N,)
            - descriptors (N, descriptor_dim)
        """
        raise NotImplementedError
