import numpy as np
import math
from sympy import solve, Symbol, latex, simplify, symbols, Matrix
import matplotlib.pyplot as plt
import seaborn as sns


def find_next_event(alpha, beta, N, n, dt):
    p_d = beta*n
    t_d = np.random.exponential(scale = 1/p_d)#*dt
    if n != N:
        p_b = alpha * n * (1 - n / N)#*dt
        #t_b = np.random.negative_binomial(n=1, p=p_b)*dt
        t_b = np.random.exponential(scale = 1/p_b)#*dt
        if t_b > t_d:
            return t_d, -1
        return t_b, 1
    return t_d, -1

alpha, beta = .012, .01
N = 100
dt = 0.001
T_ext = np.exp(N*(np.log(alpha/beta)-(1-beta/alpha)))#/dt
max_samples = 100000
n_realizations = 1000
realizations = []
r = 0
while r < n_realizations:
    n_0 = N * (1 - beta / alpha)
    n_t = [n_0]
    t_r = [0]
    sample = 0
    while sample < max_samples and n_t[-1] >= 1:
        if n_t[-1] == 0:
            break
        delta_t, event = find_next_event(alpha, beta, N, n_t[-1], dt)
        n_t.append(n_t[-1] + event)
        t_r.append(t_r[-1] + delta_t)
        sample += 1
    n_t = np.array(n_t)
    t_r = np.array(t_r)
    plt.plot(t_r, n_t)
    realizations.append(np.vstack((t_r, n_t)).T)
    r += 1

n_t_data = np.array([])
T_ext_arr = np.array([])
for realization in realizations:
    n_t_data = np.hstack((n_t_data, realization[:,1]))
    T_ext_arr = np.hstack((T_ext_arr, realization[-1,0]))

T_ext_exp = np.average(T_ext_arr)

plt.figure()
plt.hist(n_t_data, bins = 55)
plt.figure()
plt.hist(T_ext_arr, bins=50)
plt.axvline(T_ext_exp, linewidth=3, color='r', label='Numerical $\langle T_{ext} \\rangle$')
plt.axvline(T_ext, linewidth=3, color='m', linestyle='--', label='Theoretical $\langle T_{ext} \\rangle$')
plt.title(str(T_ext))
plt.legend()
plt.show()
