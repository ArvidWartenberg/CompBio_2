import numpy as np
import math
from sympy import solve, Symbol, latex, simplify, symbols, Matrix
import matplotlib.pyplot as plt
import seaborn as sns


def find_next_event(alpha, beta, N, n, dt):
    p_b = alpha * n * (1 - n/N) * dt
    p_d = beta*n*dt
    t_b = np.random.negative_binomial(n=1, p=p_b)
    t_d = np.random.negative_binomial(n=1, p=p_d)
    return min(t_b,t_d)*dt, int(t_b>=t_d) # Returns 1 if recovery happens first. 0 else.

alpha, beta = 2, 1
N = 20
dt = 0.01
n_samples = 100
n_realizations = 10
# TODO remove first sample since it is always the same, ruins statistics
cum_realizations = np.zeros((n_realizations, n_samples))
for r in range(n_realizations):
    n_0 = N * (1 - beta / alpha)
    n_t = np.zeros(n_samples)
    n_t[-1] = n_0
    for s in range(n_samples):
        t_diff, event = find_next_event(alpha, beta, N, n_t[s-1], dt)
        n_t[s] = n_t[s-1] + event - int(event==0)
    plt.plot(n_t)
    #cum_realizations[r,:] = n_t
plt.show()
plt.plot(cum_realizations)
plt.show()
