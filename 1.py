import numpy as np
from sympy import solve, Symbol, latex, simplify, symbols
import matplotlib.pyplot as plt
import seaborn as sns

# Symbolically find fixpts
'''
a,b,d,K,S,I,D = symbols('a b d K S I D')
dSdt = b*(I+S)-d*S-S*(I+S)/K-a*S*I
dIdt = -d*I-I*(S+I)/K+a*S*I
sol = solve([dSdt,dIdt],[S,I])

# Print for latex
for j in range(len(sol)):
    print('(S^*_%i,I^*_%i)&='%(j+1,j+1) + str(latex(sol[j])) + '\\\\')
    #print(sol[j])
'''

# Define Habitat class
class Habitat:

    # Constructor
    def __init__(self, S, I, L, a, b, d, K, D, h=1, gens=0, dt=0.01):
        self.S = S
        self.I = I
        self.L = L
        self.a = a
        self.b = b
        self.d = d
        self.K = K
        self.D = D
        self.h = h
        self.gens = gens
        self.dt = dt

    # Get most recent S and I
    def get_most_recent(self):
        if self.gens == 0:
            return np.copy(self.S), np.copy(self.I)
        else:
            return np.copy(self.S[self.gens,:]), np.copy(self.I[self.gens,:])

    # Iterate one time step dt
    def iterate(self, n_iters):

        # Update habitat n_iters times
        for iter in range(n_iters):

            S_current, I_current = self.get_most_recent()
            S_upd, I_upd = np.copy(S_current), np.copy(I_current)

            for i in range(L):
                # Add change due to diffusion
                S_upd[i] += self.D*(S_current[(i+1)%L] + S_current[i-1] - 2*S_current[i])/self.h**2
                I_upd[i] += self.D*(I_current[(i+1)%L] + I_current[i-1] - 2*I_current[i])/self.h**2

                # Account for other terms
                S_upd[i] += b*(S_current[i] + I_current[i]) - d*S_current[i] - S_current[i]*(S_current[i]
                                                                                 + I_current[i])/K -a*S_current[i]*I_current[i]
                I_upd[i] += - d*I_current[i] - I_current[i]*(S_current[i] + I_current[i])/K + a*S_current[i]*I_current[i]

                # Update habitat
            self.S = np.vstack([S_upd, self.S])
            self.I = np.vstack([I_upd, self.I])

            # Account for new generation
            print(self.gens)

            # Print status
            print('Finished iter ' + str(iter+1))

    def iterate_updated(self, n_iters):

        # Update habitat n_iters times
        for iter in range(n_iters):
            S_current, I_current = self.get_most_recent()
            S_dot = np.zeros(L)
            I_dot = np.zeros(L)
            if iter == 1:
                print()
            for i in range(L):
                S_dot[i] = self.b*(S_current[i] + I_current[i]) - self.d*S_current[i] \
                           - S_current[i]*(S_current[i] + I_current[i])/self.K -self.a*S_current[i]*I_current[i]\
                           + self.D*(S_current[(i+1)%self.L] + S_current[i-1] - 2*S_current[i])/self.h**2
                I_dot[i] = - self.d*I_current[i]\
                           - I_current[i]*(S_current[i] + I_current[i])/self.K\
                           + self.a*S_current[i]*I_current[i] + self.D*(I_current[(i+1)%self.L]
                           + I_current[i-1] - 2*I_current[i])/self.h**2

            # Add to S, I
            self.S = np.vstack([self.S, S_current + S_dot*self.dt])
            self.I = np.vstack([self.I, I_current + I_dot*self.dt])
            self.gens += 1
            print('Finished iter ' + str(iter+1))


    # Get total population over generations represented as matrix
    def get_N(self):
        return self.S + self.I


L = 100
a = .1
b = 1
d = .5
K = 30
D = 1
dt = .1
n_iters = 700
t = np.arange(0, n_iters + 1, 1)*dt

# Define initial conditions from intresting fixpoint
S_init = np.zeros(L)
I_init = np.zeros(L)
I_init[0], S_init[0] = b/a, (K*(a*b-a*d)-b)/a

habitat = Habitat(S=S_init, I=I_init, L=L, a=a, b=b, d=d, K=K, D=D, h=1, gens=0, dt=dt)
habitat.iterate_updated(n_iters=n_iters)
S = habitat.S
I = habitat.I
N = habitat.get_N()


plt.imshow(N, cmap='autumn', interpolation='nearest', aspect='auto')
plt.xlabel('Habitat index', fontsize=15)
plt.ylabel('Time [dt=%.2f]'%dt, fontsize=15)
plt.title('Pop. evol. for viral spread in %i habitats'%L, fontsize=17)

plt.figure()
plt.imshow(I, cmap='autumn', interpolation='nearest', aspect='auto')
plt.xlabel('Habitat index', fontsize=15)
plt.ylabel('Time [dt=%.2f]'%dt, fontsize=15)
plt.title('Infectives evol. for viral spread in %i habitats'%L, fontsize=17)

plt.figure()
plt.imshow(S, cmap='autumn', interpolation='nearest', aspect='auto')
plt.xlabel('Habitat index', fontsize=15)
plt.ylabel('Time [dt=%.2f]'%dt, fontsize=15)
plt.title('Sucept. evol. for viral spread in %i habitats'%L, fontsize=17)

plt.figure()
sns.heatmap(S-I)#, cmap='autumn', interpolation='nearest', aspect='auto')
plt.xlabel('Habitat index', fontsize=15)
plt.ylabel('Time [dt=%.2f]'%dt, fontsize=15)
plt.title('$S-I$. evol. for viral spread in %i habitats'%L, fontsize=17)

plt.figure()
plt.plot(S[:,30]-I[:,30], linewidth=3, label='$S-I$')
plt.axhline(S[n_iters-10,30], linewidth=3, linestyle='--', c='r', label='Steady state')
plt.xlabel('Time [dt]', fontsize=15)
plt.ylabel('Suceptibles - Infectives', fontsize=15)
plt.title('$S-I$ vs. time in 30th habitat', fontsize=17)
plt.show()

'''
The observed result, i.e., that the suceptibles spread faster than the infectives is expected.
The reason for this is that infectives a transmission of the disease from an infected individual
to a suceptible can one can only occur if there are suceptibles in the habitat. With this said,
let us recall that our stable fixpoint, with the chosen parameter values, has S=10, I=5.
Since both infectives and suceptibles diffuse throughout the habitats with the same rate,
it is more probable (or in this case guaranteed, since we are using a deterministic model) that
the suceptibles will outnumber the infectives when the population diffuses. 

'''


#TODO less rough, more exact
'''
We definitely see a travelling wave in the simulation results. It seems to be travelling at 
constant velocity. We can see that the wave has traveled across 50 habitats during the time 430dt.
                                        
                                        --> The velocity of the travelling wave is 50[habitat length]/430dt
                                                                                    = 0.11627 [habitat length]/[dt]
'''

