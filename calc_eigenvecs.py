from eigenvalues import *
from elliptic_integrals import *
from scipy.integrate import quad
from scipy.linalg import null_space
import numpy as np
dims = (2, 2, 1)
kapp = kappa_pm("+", dims)
alpha_pp_0 = ellip_pp_cnst(dims, "alpha")
beta_pp_0 = ellip_pp_cnst(dims, "beta")
gamma_pp_0 = ellip_pp_cnst(dims, "gamma")
d = beta_pp_0 * gamma_pp_0 + gamma_pp_0 * alpha_pp_0 + alpha_pp_0 * beta_pp_0
A = np.array(
        [
            [(kapp - 1) + (4*alpha_pp_0)/(3*d), -(2*beta_pp_0)/(3*d), -(2*gamma_pp_0)/(3 * d)],
            [-(2*alpha_pp_0)/(3*d), (kapp - 1) + (4*beta_pp_0)/(3*d), -(2*gamma_pp_0)/(3*d)],
            [1., 1., 1.]
        ]
)
print(A)
print(null_space(A))
