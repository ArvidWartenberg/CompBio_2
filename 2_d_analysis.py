import numpy as np
import math
from sympy import solve, Symbol, latex, simplify, symbols, Matrix
import matplotlib.pyplot as plt
from scipy.stats import norm
import seaborn as sns
from scipy.signal import savgol_filter

def get_rho_n_t(realizations, t_ix, delta_ix):
    data = np.squeeze(np.asarray(realizations[t_ix-delta_ix:t_ix+delta_ix, :]))
    bin_max = np.max(data)
    bins = np.arange(-.2, bin_max+1, 1)
    hist = np.histogram(data, bins=bins, density=True)
    rho = hist[0]
    #rho = savgol_filter(rho, 7, 3)
    bins = hist[1][0:len(rho)]
    mu, std = norm.fit(data)
    p = norm.pdf(bins, mu, std)
    return rho, bins, p

def get_rho_theory(alpha, beta, N, I_max):
    r_0 = alpha / beta
    I = np.linspace(0, I_max, 100)
    I_min = 1-1/r_0
    # This expression for S_0 was found analytically...
    S_0 = -(-np.log(1 - I) + I*(-1 + np.log(r_0 - r_0*I)))-(-np.log(1 - I_min) + I_min*(-1 + np.log(r_0 - r_0*I_min)))
    rho = np.exp(-N * S_0)
    n = I * N
    rho = rho/np.abs(np.trapz(n, rho))
    return n, rho


realizations = np.load('R_2.npy')
extinction_times = np.load('E_ts_2.npy')
expected_delta_t = np.load('dt_2.npy')
expected_t_ix = np.load('E_ix_2.npy')
alpha, beta = 1, 0.9
N = 100

t_1 = 10
t_2 = expected_t_ix
t_3 = int(expected_t_ix*2)
delta_ix = 1

rho_t1, bins_1, p_1 = get_rho_n_t(realizations=realizations, t_ix=t_1, delta_ix=delta_ix)
rho_t2, bins_2, p_2 = get_rho_n_t(realizations=realizations, t_ix=t_2, delta_ix=delta_ix)
rho_t3, bins_3, p_3 = get_rho_n_t(realizations=realizations, t_ix=t_3, delta_ix=delta_ix)



plt.scatter(bins_1,-np.log(rho_t1), c='b', label='Observation: $t_1=%i<T_{ext}$'%t_1, linewidth=2, alpha=.5)
plt.plot(bins_1,-np.log(p_1), c='m', label='Gaussian fit: $t_1=%i$'%t_1, linewidth=2, linestyle=':')
plt.xlabel('n infectives', fontsize=15)
plt.ylabel('$-$log$(rho_n(t))$', fontsize=15)
plt.tight_layout(True)
plt.legend()

plt.figure()
plt.scatter(bins_1,-np.log(rho_t1), marker='x', c='r', label='Observation: $t_1=%i<T_{ext}$'%t_1, linewidth=2, alpha=.5)
plt.scatter(bins_2,-np.log(rho_t2), marker='o',c='g', label='Observation: $t_2=%i$~$=T_{ext}$'%t_2, linewidth=2, alpha=.5)
plt.scatter(bins_3,-np.log(rho_t3), marker='o', c='b', label='Observation: $t_3=%i>T_{ext}$'%t_3, linewidth=2, alpha=.5)
plt.xlabel('n infectives', fontsize=15)
plt.ylabel('$-$log$(rho_n(t))$', fontsize=15)
plt.title('Logarithmic infectives density for different t', fontsize=17)
plt.tight_layout(True)
plt.legend()



T_ext = np.average(extinction_times)
n_bins=100
t_bins = (np.linspace(0,max(extinction_times), n_bins)).astype('int')
ext_bins = np.histogram(extinction_times, bins=t_bins, density=True)[0]


plt.figure()
plt.plot(t_bins[0:n_bins-1], np.cumsum(ext_bins)/np.max(np.cumsum(ext_bins)), linewidth=3, color='b', label='Extinction density')
plt.axvline(T_ext, linewidth=3, c='r', linestyle='--', label='T_ext')
plt.xlabel('Time [unit time]', fontsize=15)
plt.ylabel('Extinction density', fontsize=15)
plt.title('Extinction probability vs. time', fontsize=17)
plt.tight_layout(True)

n, rho_theory = get_rho_theory(alpha, beta, N, max(bins_3)/N)
plt.figure()
plt.scatter(bins_3,-np.log(rho_t3), marker='o', c='b', label='Observation: $t_3=%i>T_{ext}$'%t_3, linewidth=2, alpha=.5)
plt.plot(n,-np.log(rho_theory), c='r', label='Theoretical $\\rho_0$', linewidth=2, alpha=.5)
plt.xlabel('n infectives', fontsize=15)
plt.ylabel('$-$log$(rho_n(t))$', fontsize=15)
plt.title('Logarithmic infectives density for different t', fontsize=17)
plt.tight_layout(True)
plt.legend()


plt.show()
