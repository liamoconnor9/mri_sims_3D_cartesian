"""Use leading order approximation to compute growth rate

"""

import numpy as np

def calc_asymptotic_growth(R, q, B, d, ky, kz):
    Rprime = R**2 - 1
    omega2 = q*B**2/(q+1)*(kz**2*(d**2/np.pi**2 * kz**2 - Rprime) - ((6 + np.pi**2)*q + np.pi**2 - 6)*ky**2/12.)
    omega = np.sqrt(-omega2)

    return omega, omega2
