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

