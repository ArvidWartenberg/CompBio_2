import numpy as np
import math
from sympy import solve, Symbol, latex, simplify, symbols, Matrix
import matplotlib.pyplot as plt
from scipy.stats import norm
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
            return t_d, n-1
        return t_b, n+1
    return t_d, n-1

# Produce realizations using Gillepsi's algorithm
def produce_realizations(n_realizations, max_samples, alpha, beta, N):
    realizations = np.zeros((max_samples, n_realizations))
    extinction_ixs = np.zeros(n_realizations)
    expected_delta_t = 0

    n_times = 0
    r = 0
    while r < n_realizations:
        print(r)
        n = N * (1 - beta / alpha)
        sample = 0
        while sample < max_samples-1 and n>0:
            delta_t, n = find_next_event(alpha=alpha, beta=beta, N=N, n=n)
            if n<0:
                break
            realizations[sample, r] = n
            sample += 1
            n_times += 1
            expected_delta_t += delta_t


        extinction_ixs[r] = sample*int(sample < max_samples-2)
        r += 1

    expected_delta_t = expected_delta_t/n_times
    expected_t_ix = int(n_times/r)
    extinction_ixs = extinction_ixs[np.where(extinction_ixs != 0)]
    return realizations, extinction_ixs, expected_delta_t, expected_t_ix




alpha, beta = 1, .8
N = 100
max_samples = 30000
n_realizations = 1000

realizations, extinction_times, expected_delta_t, expected_t_ix = produce_realizations(n_realizations=n_realizations,
                                                                        max_samples=max_samples,
                                                                        alpha=alpha, beta=beta, N=N)

np.save('R', realizations)
np.save('E_ts', extinction_times)
np.save('dt', expected_delta_t)
np.save('E_ix', expected_t_ix)

'''
T_ext = np.average(extinction_times)
n_bins=10
t_bins = (np.linspace(0,max(extinction_times), n_bins)).astype('int')
ext_bins = np.histogram(extinction_times, bins=t_bins, normed=True)[0]
ext_bins = savgol_filter(ext_bins, 11, 3)

plt.plot(t_bins[0:n_bins-1], ext_bins, linewidth=3, color='b', label='Extinction density')
plt.axvline(T_ext, linewidth=3, c='r', linestyle='--', label='T_ext')
plt.xlabel('Time [unit time]', fontsize=15)
plt.ylabel('Extinction density', fontsize=15)
plt.title('Extinction probability vs. time', fontsize=17)
plt.tight_layout(True)
'''

'''
rho_t0, p_0, bins_0 = get_rho_n_t(realizations=realizations, t_ix=t_0, delta_ix=delta_ix, bins=bins_n)
rho_t1, p_1, bins_1 = get_rho_n_t(realizations=realizations, t_ix=t_1, delta_ix=delta_ix, bins=bins_n)
rho_t2, p_2, bins_2 = get_rho_n_t(realizations=realizations, t_ix=t_2, delta_ix=delta_ix, bins=bins_n)
rho_t3, p_3, bins_3 = get_rho_n_t(realizations=realizations, t_ix=t_3, delta_ix=delta_ix, bins=bins_n)

bins = bins_1

plt.figure()
plt.plot(bins,-np.log(rho_t0), c='m', label='$t_0$', linewidth=3)
plt.plot(bins,-np.log(rho_t1), c='r', label='$t_1$', linewidth=3)
plt.plot(bins,-np.log(rho_t2), c='g', label='$t_2$', linewidth=3)
plt.plot(bins,-np.log(rho_t3), c='b', label='$t_3$', linewidth=3)
plt.plot(bins,-np.log(p_0), c='m', label='$t_0$', linewidth=2, linestyle=':')
plt.plot(bins,-np.log(p_1), c='r', label='$t_1$', linewidth=2, linestyle=':')
plt.plot(bins,-np.log(p_2), c='g', label='$t_2$', linewidth=2, linestyle=':')
plt.plot(bins,-np.log(p_3), c='b', label='$t_3$', linewidth=2, linestyle=':')
plt.xlabel('n infected', fontsize=15)
plt.ylabel('$-log(\\rho_n(t))$', fontsize=15)
plt.title('Logarithmic density for different t', fontsize=17)
plt.legend()
plt.tight_layout(True)

plt.figure()
plt.plot(bins,rho_t0, c='m', label='$t_0$', linewidth=2)
plt.plot(bins,rho_t1, c='r', label='$t_1$', linewidth=2)
plt.plot(bins,rho_t2, c='g', label='$t_2$', linewidth=2)
plt.plot(bins,rho_t3, c='b', label='$t_3$', linewidth=2)
plt.plot(bins,p_0, c='m', label='$t_0$', linewidth=2, linestyle=':')
plt.plot(bins,p_1, c='r', label='$t_1$', linewidth=2, linestyle=':')
plt.plot(bins,p_2, c='g', label='$t_2$', linewidth=2, linestyle=':')
plt.plot(bins,p_3, c='b', label='$t_3$', linewidth=2, linestyle=':')
plt.xlabel('n infected', fontsize=15)
plt.ylabel('$\\rho_n(t)$', fontsize=15)
plt.title('Density for different t', fontsize=17)
plt.tight_layout(True)

plt.figure()
plt.plot(bins,np.cumsum(rho_t0), c='m', label='$t_0$', linewidth=2)
plt.plot(bins,np.cumsum(rho_t1), c='r', label='$t_1$', linewidth=2)
plt.plot(bins,np.cumsum(rho_t2), c='g', label='$t_2$', linewidth=2)
plt.plot(bins,np.cumsum(rho_t3), c='b', label='$t_3$', linewidth=2)
plt.plot(bins,np.cumsum(p_0), c='m', label='$t_0$', linewidth=2, linestyle=':')
plt.plot(bins,np.cumsum(p_1), c='r', label='$t_1$', linewidth=2, linestyle=':')
plt.plot(bins,np.cumsum(p_2), c='g', label='$t_2$', linewidth=2, linestyle=':')
plt.plot(bins,np.cumsum(p_3), c='b', label='$t_3$', linewidth=2, linestyle=':')
plt.xlabel('n infected', fontsize=15)
plt.ylabel('$Cumulative density$', fontsize=15)
plt.title('Cumulative density for different t', fontsize=17)
plt.tight_layout(True)

plt.legend()
plt.show()
print()
'''


