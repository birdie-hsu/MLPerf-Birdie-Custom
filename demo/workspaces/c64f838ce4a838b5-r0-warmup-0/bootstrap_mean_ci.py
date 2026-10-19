import random
import math


def bootstrap_mean_ci(data, n_iterations=500, ci_level=0.95, seed=None):
    x = list(data)
    rng = random.Random(seed)
    means = sorted(sum(rng.choices(x, k=len(x))) / len(x)
                   for _ in range(n_iterations))
    # BUG: confidence level was confused with an upper quantile.
    return sum(x) / len(x), means[0], means[int(ci_level * (len(means) - 1))]
