import numpy as np
from sympy import solve, Symbol, latex, simplify, symbols, Matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Symbolically find fixpts

a,b,d,K,S,I,D = symbols('a b d K S I D')
dSdt = b*(I+S)-d*S-S*(I+S)/K-a*S*I
dIdt = -d*I-I*(S+I)/K+a*S*I
sol = solve([dSdt,dIdt],[S,I])


# Print for latex
for j in range(len(sol)):
    print('(S^*_%i,I^*_%i)&='%(j+1,j+1) + str(latex(sol[j])) + '\\\\')

# Biologically relevant soln.
sol_bio = sol[2]

# Get Jacobian
X = Matrix([S, I])
Y = Matrix([dSdt, dIdt])
J = Y.jacobian(X)
print(latex(simplify(J)))

# Get eigs
eigs = list(J.eigenvals().keys())
eig_1 = simplify(eigs[0].subs([(S, sol_bio[0]), (I, sol_bio[1])]))
eig_2 = simplify(eigs[1].subs([(S, sol_bio[0]), (I, sol_bio[1])]))

#print eigs
print(latex(eig_1))
print(latex(eig_2))