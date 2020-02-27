import numpy as np
import math
from sympy import solve, Symbol, latex, simplify, symbols, Matrix
import matplotlib.pyplot as plt
from scipy.stats import norm
import seaborn as sns
from scipy.signal import savgol_filter

def get_rho_n_t(realizations, t_ix, delta_ix, n_bins):
    data = np.squeeze(np.asarray(realizations[t_ix-delta_ix:t_ix+delta_ix, :]))
    data = data[np.where(data!=0)]
    #bin_max = 23 #np.max(data)-3
    #bin_min = np.min(data)-1
    #bins = np.arange(bin_m, bin_max, 1.5)
    hist = np.histogram(data, bins=10, density=True)
    rho = hist[0]
    bins = hist[1][0:len(rho)]
    mu, std = norm.fit(data)
    p = norm.pdf(bins, mu, std)
    return rho, bins, p

realizations = np.load('R')
extinction_times = np.load('E_ts')
expected_delta_t = np.load('dt')
expected_t_ix = np.load('E_ix')

t_0 = 10
t_1 = int(expected_t_ix/7)
t_2 = expected_t_ix
t_3 = int(expected_t_ix*2)
delta_ix = 1
n_bins = 20

'''
rho_t0, p_0, bins_0 = get_rho_n_t(realizations=realizations, t_ix=t_0, delta_ix=delta_ix, n_bins=n_bins)
rho_t1, p_1, bins_1 = get_rho_n_t(realizations=realizations, t_ix=t_1, delta_ix=delta_ix, n_bins=n_bins)
rho_t2, p_2, bins_2 = get_rho_n_t(realizations=realizations, t_ix=t_2, delta_ix=delta_ix, n_bins=n_bins)
rho_t3, p_3, bins_3 = get_rho_n_t(realizations=realizations, t_ix=t_3, delta_ix=delta_ix, n_bins=n_bins)
'''
rho_t0, bins_0, p_0 = get_rho_n_t(realizations=realizations, t_ix=t_0, delta_ix=delta_ix, n_bins=n_bins)
rho_t1, bins_1, p_1 = get_rho_n_t(realizations=realizations, t_ix=t_1, delta_ix=delta_ix, n_bins=n_bins)
rho_t2, bins_2, p_2 = get_rho_n_t(realizations=realizations, t_ix=t_2, delta_ix=delta_ix, n_bins=n_bins)
rho_t3, bins_3, p_3 = get_rho_n_t(realizations=realizations, t_ix=t_3, delta_ix=delta_ix, n_bins=n_bins)


plt.subplot(2,2,1)
plt.plot(bins_0,-np.log(rho_t0), c='b', label='$Observation: t_0=%i$'%t_0, linewidth=3)
plt.plot(bins_0,-np.log(p_0), c='m', label='$Gaussian fit: t_0=%i$'%t_0, linewidth=2, linestyle=':')
plt.xlabel('n infectives', fontsize=15)
plt.ylabel('$-$log$(rho_n(t))$', fontsize=15)
plt.legend()
plt.subplot(2,2,2)
plt.plot(bins_1,-np.log(rho_t1), c='b', label='$Observation: t_1=%i$'%t_1, linewidth=3)
plt.plot(bins_1,-np.log(p_1), c='m', label='$Gaussian fit: t_1=%i$'%t_1, linewidth=2, linestyle=':')
plt.xlabel('n infectives', fontsize=15)
plt.ylabel('$-$log$(rho_n(t))$', fontsize=15)
plt.legend()
plt.subplot(2,2,3)
plt.plot(bins_2,-np.log(rho_t2), c='b', label='$Observation: t_2=%i$'%t_2, linewidth=3)
plt.plot(bins_2,-np.log(p_2), c='m', label='$Gaussian fit: t_2=%i$'%t_2, linewidth=2, linestyle=':')
plt.xlabel('n infectives', fontsize=15)
plt.ylabel('$-$log$(rho_n(t))$', fontsize=15)
plt.legend()
plt.subplot(2,2,4)
plt.plot(bins_3,-np.log(rho_t3), c='b', label='$Observation: t_3=%i$'%t_3, linewidth=3)
plt.plot(bins_3,-np.log(p_3), c='m', label='$Gaussian fit: t_3=%i$'%t_3, linewidth=2, linestyle=':')
plt.xlabel('n infectives', fontsize=15)
plt.ylabel('$-$log$(rho_n(t))$', fontsize=15)
plt.legend()
plt.show()
