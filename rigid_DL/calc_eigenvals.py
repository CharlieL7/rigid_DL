from rigid_DL.eigenvalues import *
from rigid_DL.elliptic_integrals import *
from scipy.integrate import quad
from scipy.linalg import null_space
import ast
import numpy as np
dims = ast.literal_eval(input("dimensions as (a, b, c): "))
kapp = kappa_pm("+", dims)
print("lambda_12 eigenvalue {}".format(lambda_12(dims)))
print("lambda_23 eigenvalue {}".format(lambda_23(dims)))
print("lambda_31 eigenvalue {}".format(lambda_31(dims)))
print("lambda_p eigenvalue {}".format(lambda_pm("+", dims)))
print("lambda_m eigenvalue {}".format(lambda_pm("-", dims)))
