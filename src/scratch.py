#
# from sklearn.model_selection import ParameterGrid
# from scipy.optimize import brute, minimize
# import numpy as np
#
# def quadratic(x: float, A: float, B: float, C: float) -> float:
#     """Quadratic equation y-value from Ax^2 + Bx + C with specific x-value plugged in"""
#     return A * x**2 + B * x + C
#
# x0, fval, grid, Jout = brute(func=quadratic, ranges=((0, 9),), Ns=10, args=(1, -6, 8), full_output=True, finish=None)
# print(f'x0 = {x0}')
# print(f'fval = {fval}')
# print(f'grid = {grid}')
# print(f'Jout = {Jout}')
#
# result = minimize(fun=quadratic, method='Nelder-Mead', x0=1, args=(1, -6, 8), tol=0.1)
# print(result)

import numpy as np

ans = np.random.choice(a=[1, 2, 3, 4], p=[0.1, 0.2, 0.5, 0.001])
print(ans)