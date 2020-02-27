import numpy as np
import math
from sympy import solve, Symbol, latex, simplify, symbols, Matrix
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import savgol_filter

# Simulate time to next event and time using exponential distributions
def find_next_event(alpha, beta, N, n):
    p_d = beta*n
    t_d = np.random.exponential(scale = 1/p_d)
    if n != N:
        p_b = alpha * n * (1 - n / N)
        t_b = np.random.exponential(scale = 1/p_b)
        if t_b > t_d:
            return t_d, -1
        return t_b, 1
    return t_d, -1

# Produce realizations using Gillepsi's algorithm
def produce_realizations(n_realizations, max_samples, alpha, beta, N):
    realizations = np.zeros((max_samples, n_realizations))
    extinction_times = np.zeros(n_realizations)
    expected_delta_t = 0
    n_times = 0
    r = 0
    while r < n_realizations:
        print(r)
        n_0 = N * (1 - beta / alpha)
        n_t = [n_0]
        t_r = [0]
        sample = 0
        while sample < max_samples-1 and n_t[-1] >= 1:
            if n_t[-1] == 0:
                break
            delta_t, event = find_next_event(alpha=alpha, beta=beta, N=N, n=n_t[-1])
            n_t.append(n_t[-1] + event)
            t_r.append(t_r[-1] + delta_t)
            sample += 1
            n_times += 1
            expected_delta_t += delta_t
        n_t = np.array(n_t)
        t_r = np.array(t_r)
        realizations[0:len(n_t),r] = n_t
        extinction_times[r] = t_r[-1]*int(len(t_r)<max_samples-1)
        r += 1
    expected_delta_t = expected_delta_t/n_times
    extinction_times = extinction_times[np.where(extinction_times != 0)]
    return realizations, extinction_times, expected_delta_t

def get_rho_n_t(realizations, t_ix, delta_ix, bins = np.arange(0,50,1)):
    data = realizations[t_ix-delta_ix:t_ix+delta_ix, :]
    #data[np.where(data==0)] = np.nan
    rho_n_t = np.histogram(realizations[t_ix-delta_ix:t_ix+delta_ix, :], bins=bins, density=True)[0]
    rho_n_t_smooth = savgol_filter(rho_n_t, 11, 3)
    return rho_n_t_smooth #rho_n_t_smooth


alpha, beta = .012, .01
N = 100
dt = 0.001
max_samples = 3000
n_realizations = 10000

realizations, extinction_times, expected_delta_t = produce_realizations(n_realizations=n_realizations,
                                                                        max_samples=max_samples,
                                                                        alpha=alpha, beta=beta, N=N)
T_ext = np.average(extinction_times)
n_bins=50
t_bins = (np.linspace(0,max(extinction_times), n_bins)).astype('int')
ext_bins = np.histogram(extinction_times, bins=t_bins, normed=True)[0]
ext_bins = savgol_filter(ext_bins, 11, 3)
plt.plot(t_bins[0:n_bins-1], ext_bins, linewidth=3, color='b', label='Extinction density')
plt.axvline(T_ext, linewidth=3, c='r', linestyle='--', label='T_ext')
plt.xlabel('Time', fontsize=15)
plt.ylabel('Extinction density', fontsize=15)
plt.title('Extinction probability vs. time', fontsize=17)

t_0 = 20
t_1 = int(T_ext/6/expected_delta_t)
t_2 = int(T_ext/expected_delta_t)
t_3 = int(T_ext*(1+5/6)/expected_delta_t)
delta_ix = 10
bins_n = np.arange(0,50,1)
rho_t0 = get_rho_n_t(realizations=realizations, t_ix=t_0, delta_ix=delta_ix, bins=bins_n)
rho_t1 = get_rho_n_t(realizations=realizations, t_ix=t_1, delta_ix=delta_ix, bins=bins_n)
rho_t2 = get_rho_n_t(realizations=realizations, t_ix=t_2, delta_ix=delta_ix, bins=bins_n)
rho_t3 = get_rho_n_t(realizations=realizations, t_ix=t_3, delta_ix=delta_ix, bins=bins_n)

plt.figure()
plt.plot(-np.log(rho_t0), c='m', label='$t_1$', linewidth=3)
plt.plot(-np.log(rho_t1), c='r', label='$t_1$', linewidth=3)
plt.plot(-np.log(rho_t2), c='g', label='$t_2$', linewidth=3)
plt.plot(-np.log(rho_t3), c='b', label='$t_3$', linewidth=3)
plt.legend()

plt.figure()
plt.plot(rho_t0, c='m', label='$t_0$', linewidth=2)
plt.plot(rho_t1, c='r', label='$t_1$', linewidth=2)
plt.plot(rho_t2, c='g', label='$t_2$', linewidth=2)
plt.plot(rho_t3, c='b', label='$t_3$', linewidth=2)

plt.title('blabla')
plt.xlabel('n infected', fontsize=15)
plt.xlabel('n infected', fontsize=15)
plt.legend()
plt.show()
print()



