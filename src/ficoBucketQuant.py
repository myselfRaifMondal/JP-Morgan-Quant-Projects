"""Task 4 - Bucket FICO scores into ratings (JPMorgan Quantitative Research, Forage).

Provides two ways of splitting a set of FICO scores into ``num_buckets``
contiguous rating buckets:

* ``mse_bucketization`` - sorts the scores and cuts them into equally sized
  buckets, returning ``(low_score, high_score, bucket_mean)`` per bucket.
* ``optimize_log_likelihood`` - intended to start from evenly spaced boundaries
  between the minimum and maximum score and use ``scipy.optimize.minimize``
  (Powell) to move the interior boundaries so that the bucket log-likelihood of
  the observed defaults is maximised, returning the optimised
  ``(lower_bound, upper_bound)`` pairs.

  KNOWN BUG: it does not currently run. ``optimize_log_likelihood`` passes
  ``defaults`` keyed by FICO score into ``log_likelihood``, which indexes it by
  ``(lower_bound, upper_bound)`` bucket tuple, so the first objective
  evaluation raises ``KeyError``. ``log_likelihood`` expects ``defaults`` and
  ``total_records`` to be dicts keyed by the same bucket objects it iterates
  over.

Input: this script does NOT read ``datasets/customerloan.csv``. The block at
the bottom of the file generates 1000 random FICO scores in [300, 850) with
``numpy.random.randint`` and a dictionary of random default counts, so results
differ on every run. To use the real data, feed the ``fico_score`` and
``default`` columns of ``datasets/customerloan.csv`` into the two functions
instead.

Output: prints the MSE buckets, then attempts to print the log-likelihood
buckets - which fails with the ``KeyError`` described above.
"""

import numpy as np
import scipy.optimize as opt

def mse_bucketization(scores, num_buckets):
    """Bucketize scores by minimizing Mean Squared Error (MSE)"""
    scores = np.sort(scores)
    n = len(scores)
    bucket_size = n // num_buckets
    buckets = []
    
    for i in range(num_buckets):
        if i == num_buckets - 1:
            bucket = scores[i * bucket_size:]
        else:
            bucket = scores[i * bucket_size:(i + 1) * bucket_size]
        
        bucket_mean = np.mean(bucket)
        buckets.append((bucket[0], bucket[-1], bucket_mean))
    
    return buckets

def log_likelihood(buckets, defaults, total_records):
    """Compute Log-Likelihood for given buckets."""
    ll = 0
    for bucket in buckets:
        ni = total_records[bucket]
        ki = defaults[bucket]
        pi = ki / ni if ni > 0 else 0
        if pi > 0 and pi < 1:
            ll += ki * np.log(pi) + (ni - ki) * np.log(1 - pi)
    return ll

def optimize_log_likelihood(scores, defaults, num_buckets):
    """Optimize buckets to maximize log-likelihood."""
    scores = np.sort(scores)
    n = len(scores)
    bucket_bounds = np.linspace(scores[0], scores[-1], num_buckets + 1)
    
    def objective(bounds):
        bounds = np.sort(bounds)
        buckets = [(bounds[i], bounds[i+1]) for i in range(len(bounds) - 1)]
        return -log_likelihood(buckets, defaults, {b: np.sum((scores >= b[0]) & (scores < b[1])) for b in buckets})
    
    result = opt.minimize(objective, bucket_bounds[1:-1], method='Powell')
    optimized_bounds = np.sort(result.x)
    
    return [(optimized_bounds[i], optimized_bounds[i+1]) for i in range(len(optimized_bounds) - 1)]

# Example usage:
scores = np.random.randint(300, 850, 1000)  # FICO scores
defaults = {s: np.random.randint(0, 10) for s in scores}  # Simulated defaults

buckets_mse = mse_bucketization(scores, 5)
buckets_ll = optimize_log_likelihood(scores, defaults, 5)

print("Buckets using MSE:", buckets_mse)
print("Buckets using Log-Likelihood:", buckets_ll)

