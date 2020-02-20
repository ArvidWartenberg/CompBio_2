import numpy as np
from sympy import solve, Symbol, latex, simplify, symbols
import matplotlib.pyplot as plt

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
    def __init__(self, S, I, a, b, d, K, D):
        self.S = S
        self.I = I
        self.a = a
        self.b = b
        self.d = d
        self.K = K
        self.D = D

    # Iterate one time step dt
    def iterate(self):
        # Start by calculating d^2S/dx^2 using symmetric derivative
        print()


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
print()
