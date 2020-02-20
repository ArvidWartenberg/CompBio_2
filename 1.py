import numpy as np
from sympy import solve, Symbol, latex, simplify, symbols
import matplotlib.pyplot as plt
#i = Symbol('i')
#s = Symbol('s')
#a = Symbol('a')
#b = Symbol('b')
#d = Symbol('d')
#k = Symbol('k')
a,b,d,K,S,I,D = symbols('a b d K S I D')
dSdt = b*(I+S)-d*S-S*(I+S)/K-a*S*I
dIdt = -d*I-I*(S+I)/K+a*S*I
#sol1 = solve([dIdt,dSdt],[S,I])
#eq1 = S+1
#eq2 = I-1
sol = solve([dSdt,dIdt],[S,I])
for j in range(len(sol)):
    print('(S^*_%i,I^*_%i)&='%(j+1,j+1) + str(latex(sol[j])) + '\\\\')
    #print(sol[j])

class habitat:

    def __init__(self, S, I, a, b, d, K, D):
        self.S = S
        self.I = I
        self.a = a
        self.b = b
        self.d = d
        self.K = K
        self.D = D
