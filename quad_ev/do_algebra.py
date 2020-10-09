"""
Symbolic algebra code to compute the eigenvalues of the DL operator
for the 3x3 quadratic flow field system
"""

import sympy as sp

kappa = sp.symbols("kappa")
a, b, c = sp.symbols("a b c")
K_12 = sp.symbols("K_12")
K_23 = sp.symbols("K_23")
K_13 = sp.symbols("K_13")
K_123 = sp.symbols("K_123")
A_1, A_2, A_3 = sp.symbols("A_1 A_2 A_3")
H_1, H_2, H_3 = sp.symbols("H_1 H_2 H_3")

# relation between 3x3 flow matrix and the Stokes quadrupoles
hexpr_1 = A_1 * (K_23 + a**2 * K_123) + A_2 * (-K_13 + b**2 * K_123) + A_3 * (-K_12 + c**2 * K_123)
hexpr_2 = A_1 * (-K_23 + a**2 * K_123) + A_2 * (K_13 + b**2 * K_123) + A_3 * (-K_12 + c**2 * K_123)
hexpr_3 = A_1 * (-K_23 + a**2 * K_123) + A_2 * (-K_13 + b**2 * K_123) + A_3 * (K_12 + c**2 * K_123)

# the system for calculating the eigenvalue
ev_1 = (1 - kappa) * ((b**2 + c**2) * H_1 + c**2 * H_2 + b**2 * H_3) - (4 * A_1 / (a * b * c))
ev_2 = (1 - kappa) * (c**2 * H_1 + (a**2 + c**2) * H_2 + a**2 * H_3) - (4 * A_2 / (a * b * c))
ev_3 = (1 - kappa) * (b**2 * H_1 + a**2 * H_2 + (a**2 + b**2) * H_3) - (4 * A_3 / (a * b * c))

# hexpr equations subsituted in the ev equations
evh_1 = ev_1.subs({H_1: hexpr_1, H_2: hexpr_2, H_3: hexpr_3})
evh_2 = ev_2.subs({H_1: hexpr_1, H_2: hexpr_2, H_3: hexpr_3})
evh_3 = ev_3.subs({H_1: hexpr_1, H_2: hexpr_2, H_3: hexpr_3})

# row vectors of the B matrix from the B @ A = 0 sytem
b_1 = sp.collect(sp.expand(evh_1), [A_1, A_2, A_3])
b_2 = sp.collect(sp.expand(evh_2), [A_1, A_2, A_3])
b_3 = sp.collect(sp.expand(evh_3), [A_1, A_2, A_3])

B = sp.Matrix([
    [b_1.coeff(A_1), b_1.coeff(A_2), b_1.coeff(A_3)],
    [b_2.coeff(A_1), b_2.coeff(A_2), b_2.coeff(A_3)],
    [b_3.coeff(A_1), b_3.coeff(A_2), b_3.coeff(A_3)],
])

#print(sp.simplify(B))
#print(B.det())

# cubic equation to solve for the eigenvalues
B_det = sp.poly(B.det(), kappa)
#sp.pprint(B_det)
tmp = B_det.coeffs()
print(tmp)
#print(sp.solve(B_det, kappa))

# sphere test
k_cube = B_det.subs({a: 1, b: 1, c: 1, K_12: 2/5, K_23: 2/5, K_13:2/5, K_123: 2/7})
#sp.pprint(k_cube)
#sp.pprint(sp.solve(k_cube))

# inverting the hexpr relations
C = sp.Matrix([
    [hexpr_1.coeff(A_1), hexpr_1.coeff(A_2), hexpr_1.coeff(A_3)],
    [hexpr_2.coeff(A_1), hexpr_2.coeff(A_2), hexpr_2.coeff(A_3)],
    [hexpr_3.coeff(A_1), hexpr_3.coeff(A_2), hexpr_3.coeff(A_3)],
])

#print(sp.simplify(C))
C_inv = C.inv()
#sp.pprint(C_inv.subs({a: 1, b: 1, c: 1, K_12: 2/5, K_23: 2/5, K_13:2/5, K_123: 2/7, kappa: 3/5}))

#sp.pprint(B.subs({a: 1, b: 1, c: 1, K_12: 2/5, K_23: 2/5, K_13:2/5, K_123: 2/7, kappa: 3/5}))

H_vec = sp.Matrix([
    [H_1],
    [H_2],
    [H_3],
])

evec_sys = B * C_inv
#print(evec_sys)
tmp = evec_sys.subs({a: 1, b: 1, c: 1, K_12: 2/5, K_23: 2/5, K_13:2/5, K_123: 2/7, kappa: 3/35})
#sp.pprint(tmp)
#sp.pprint(tmp.nullspace())
