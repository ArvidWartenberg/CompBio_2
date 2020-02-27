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
#T_ext = 3650 #np.exp(N*(np.log(alpha/beta)-(1-beta/alpha)))
max_samples = 100000
n_realizations = 2000
realizations = []
r = 0
while r < n_realizations:
    print(r)
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
    #plt.plot(t_r, n_t)
    realizations.append(np.vstack((t_r, n_t)).T)
    r += 1

n_t_data = np.array([])
T_ext_arr = np.array([])
n_t_pre_ext = np.array([])
for realization in realizations:
    if len(realization[:,0]) < max_samples:
        T_ext_arr = np.hstack((T_ext_arr, realization[-1,0]))

T_ext_exp = int(np.average(T_ext_arr))

t_1 = int(T_ext_exp/2)
t_2 = T_ext_exp
t_3 = int(T_ext_exp*3/2)
sample_t = int(T_ext_exp/5)

rho_t = []
for t in [t_1, t_2, t_3]:
    t_data = []
    for t_prime in range(t-100,t+100,1):
        for r in range(n_realizations):
            if len(realizations[r][:,1]) > t_prime:
                t_data.append(realizations[r][t_prime,1])
    rho_t.append(np.array(t_data))

print()


'''
data_t_1 = np.array([])
data_t_2 = np.array([])
data_t_3 = np.array([])
# Collect data
for realization in realizations:
    t_len = len(realization[:,1])
    if t_len < t_1+sample_t+1:
        data_t_1 = np.hstack((data_t_1, realization[t_1-sample_t:t_1+sample_t,1]))
    if t_len < t_2 + sample_t + 1:
        data_t_2 = np.hstack((data_t_2, realization[t_2 - sample_t:t_2 + sample_t, 1]))
    if t_len < t_3 + sample_t + 1:
        data_t_3 = np.hstack((data_t_3, realization[t_3 - sample_t:t_3 + sample_t, 1]))

    n_t_pre_ext = np.hstack((n_t_pre_ext, realization[:, 1]))


plt.figure()
plt.hist(data_t_1, bins = 40, alpha=.4, color='r', density=True, log=True, label='t1')
#plt.title('t1')
#plt.figure()
plt.hist(data_t_2, bins = 40, alpha=.4, color='g', density=True, log=True, label='t2')
#plt.title('t1')
#plt.figure()
plt.hist(data_t_3, bins = 40, alpha=.4, color='b', density=True, log=True, label='t3')
#plt.title('t1')'''
plt.figure()
plt.hist(T_ext_arr, bins=40)
plt.axvline(T_ext_exp, linewidth=3, color='r', label='Numerical $\langle T_{ext} \\rangle$')
#plt.axvline(T_ext, linewidth=3, color='m', linestyle='--', label='Theoretical $\langle T_{ext} \\rangle$')
#plt.title(str(T_ext))
plt.legend()
plt.show()