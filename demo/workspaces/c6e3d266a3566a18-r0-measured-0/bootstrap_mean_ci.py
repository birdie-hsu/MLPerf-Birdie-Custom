import random
import math


def bootstrap_mean_ci(data, n_iterations=500, ci_level=0.95, seed=None):
    x = list(data)
    if not x or any(not isinstance(v, (int, float)) or isinstance(v, bool)
                    or not math.isfinite(v) for v in x):
        raise ValueError("data must contain finite numbers")
    if not isinstance(n_iterations, int) or isinstance(n_iterations, bool) or n_iterations < 1:
        raise ValueError("n_iterations must be a positive integer")
    if not isinstance(ci_level, (int, float)) or not 0 < ci_level < 1:
        raise ValueError("ci_level must be between zero and one")
    rng = random.Random(seed)
    means = sorted(sum(rng.choices(x, k=len(x))) / len(x)
                   for _ in range(n_iterations))
    def percentile(p):
        position = p * (len(means) - 1)
        lo = int(position)
        hi = min(lo + 1, len(means) - 1)
        return means[lo] + (position - lo) * (means[hi] - means[lo])
    tail = (1 - ci_level) / 2
    return sum(x) / len(x), percentile(tail), percentile(1 - tail)
