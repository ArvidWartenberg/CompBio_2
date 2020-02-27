import numpy as np
import math
from sympy import solve, Symbol, latex, simplify, symbols, Matrix
import matplotlib.pyplot as plt
import seaborn as sns



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


print()


# Print for latex
#for j in range(len(sol)):
    #print('(S^*_%i,I^*_%i)&='%(j+1,j+1) + str(latex(sol[j])) + '\\\\')
#    print(sol[j])
