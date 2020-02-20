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
    def __init__(self, S, I, a, b, d, K, D, h=1, gens=0):
        self.S = S
        self.I = I
        self.a = a
        self.b = b
        self.d = d
        self.K = K
        self.D = D
        self.h = h
        self.gens = gens

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
                self.S = np.vstack([self.S, S_upd])
                self.I = np.vstack([self.I, I_upd])

            # Account for new generation
            self.gens += 1

            # Print status
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

# Define initial conditions from intresting fixpoint
S_init = np.zeros(L)
I_init = np.zeros(L)
I_init[0], S_init[0] = b/a, (K*(a*b-a*d)-b)/a

habitat = Habitat(S=S_init, I=I_init, a=a, b=b, d=d, K=K, D=D, h=1, gens=1)
habitat.iterate(10)
N = habitat.get_N()

sns.heatmap(N.T)
plt.show()

print()
