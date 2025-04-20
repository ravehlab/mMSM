import numpy as np

def dirichlet_MMSE(transitions, alpha):
        MMSE_ids = np.array(list(transitions.keys()), dtype=int)
        MMSE_counts = np.array(list(transitions.values())) + alpha
        MMSE_counts = MMSE_counts/np.sum(MMSE_counts)
        return (MMSE_ids, MMSE_counts)
