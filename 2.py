import numpy as np
import math
from sympy import solve, Symbol, latex, simplify, symbols, Matrix
import matplotlib.pyplot as plt
import seaborn as sns



'''
# Symbolically find fixpts
alpha, beta, S, I, N = symbols('alpha beta S I N')
dSdt = -alpha*S*I/(I+S) + beta*I
dIdt = alpha*S*I/(I+S) - beta*I
dIdt_simple = alpha*(N-I)*I/N - beta*I
sol = solve([dSdt,dIdt],[S,I])
sol_2 = solve(dIdt_simple, I)
print(sol)
print(sol_2)

# Bio relevant fp
sol_bio = sol[1]
print(sol_bio)

# Get Jacobian
X = Matrix([S, I])
Y = Matrix([dSdt, dIdt])
J = Y.jacobian(X)

# Get eigs
eigs = list(J.eigenvals().keys())
print('eigs' + str(eigs))

# Linear dependence in eqts, use relevant eig..
print('eigs2' + str(simplify(eigs[0].subs([(S,sol_bio[0]),(I,sol_bio[1])]))))

# Print for latex
#for j in range(len(sol)):
    #print('(S^*_%i,I^*_%i)&='%(j+1,j+1) + str(latex(sol[j])) + '\\\\')
#    print(sol[j])
'''

def find_next_event(b_n, d_n, dt):
    t = 0
    win = -1
    while True:
        rand_1 = np.random.uniform(0, 1)
        rand_2 = np.random.uniform(0, 1)
        if rand_1 < d_n*dt:
            return t, 0
        elif rand_2 < b_n*dt:
            return t, 1
        t += dt
'''
    p_b = b_n * dt
    p_d = d_n * dt
    t_b = np.random.negative_binomial(n=1, p=p_b)*dt
    t_d = np.random.negative_binomial(n=1, p=p_d)*dt
    return min(t_b,t_d), int(t_b>=t_d) # Returns 1 if recovery happens first. 0 else.
'''

b_n, d_n =  [0.1, 1, 10], [0.2, 2, 5] # b_n: new infection, d_n: recovery
dt = 0.001 # big time steps cause problems in iterate method for last case as both should pass often but only first does
n_events = 10000

events = np.zeros(shape=[n_events, 6])
for i in range(n_events):
    events[i,0:2] = find_next_event(b_n=b_n[0], d_n=d_n[0], dt=dt)
    events[i,2:4] = find_next_event(b_n=b_n[1], d_n=d_n[1], dt=dt)
    events[i,4::] = find_next_event(b_n=b_n[2], d_n=d_n[2], dt=dt)

# Only one set of events for b_n,d_n here!
def find_event_distance_time(events):
    cum_time = np.cumsum(events[:,0])
    rec_ix = np.where(events[:,1] == 1)[0]
    inf_ix = np.where(events[:,1] == 0)[0]
    rec_ix_roll, inf_ix_roll = np.roll(rec_ix, 1), np.roll(inf_ix, 1)
    rec_times = (cum_time[rec_ix]-cum_time[rec_ix_roll])[1::]
    inf_times = (cum_time[inf_ix]-cum_time[inf_ix_roll])[1::]

    return rec_times, inf_times


rec_times_1, inf_times_1 = find_event_distance_time(events[:,0:2])
rec_times_2, inf_times_2 = find_event_distance_time(events[:,2:4])
rec_times_3, inf_times_3 = find_event_distance_time(events[:,4:6])
bins_1 = np.linspace(0, max(max(rec_times_1), max(inf_times_1)), 100)
bins_2 = np.linspace(0, max(max(rec_times_2), max(inf_times_2)), 100)
bins_3 = np.linspace(0, max(max(rec_times_3), max(inf_times_3)), 100)
line_1_b = b_n[0]*np.exp(-b_n[0]*bins_1)
line_1_d = d_n[0]*np.exp(-d_n[0]*bins_1)
line_2_b = b_n[1]*np.exp(-b_n[1]*bins_2)
line_2_d = d_n[1]*np.exp(-d_n[1]*bins_2)
line_3_b = b_n[2]*np.exp(-b_n[2]*bins_3)
line_3_d = d_n[2]*np.exp(-d_n[2]*bins_3)

plt.figure()

plt.subplot(3,1,1)
plt.title('Norm. log. Hist for time between inf/rec', fontsize=17)
plt.hist(rec_times_1, bins=bins_1, color='r', alpha=.5, label='$t_{rec}$, $b_n=.1, d_n=.2$', density=True)#, log=True)
plt.hist(inf_times_1, bins=bins_1, color='b', alpha=.5, label='$t_{inf}$, $b_n=.1, d_n=.2$', density=True)#, log=True)
plt.plot(bins_1, line_1_b, color='b', linewidth=3)
plt.plot(bins_1, line_1_d, color='r', linewidth=3)
plt.semilogy()
plt.ylabel('log(density)')
plt.legend(fontsize=10)

plt.subplot(3,1,2)
plt.hist(rec_times_2, bins=bins_2, color='r', alpha=.5, label='$t_{rec}$, $b_n=1, d_n=2$', density=True)#, log=True)
plt.hist(inf_times_2, bins=bins_2, color='b', alpha=.5, label='$t_{inf}$, $b_n=1, d_n=2$', density=True)#, log=True)
plt.plot(bins_2, line_2_b, color='b', linewidth=3)
plt.plot(bins_2, line_2_d, color='r', linewidth=3)
plt.semilogy()
plt.ylabel('log(density)')
plt.legend(fontsize=10)

plt.subplot(3,1,3)
plt.hist(rec_times_3, bins=bins_3, color='r', alpha=.5, label='$t_{rec}$, $b_n=10, d_n=5$', density=True)#, log=True)
plt.hist(inf_times_3, bins=bins_3, color='b', alpha=.5, label='$t_{inf}$, $b_n=10, d_n=5$', density=True)#, log=True)
plt.plot(bins_3, line_3_b, color='b', linewidth=3)
plt.plot(bins_3, line_3_d, color='r', linewidth=3)
plt.semilogy()
plt.xlabel('Time')
plt.ylabel('log(density)')
plt.legend(fontsize=10)
plt.show()

print()