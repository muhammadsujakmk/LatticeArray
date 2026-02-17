import numpy as np
from scipy.constants import epsilon_0, mu_0, c
import matplotlib.pyplot as plt
from em_pol_func import alpha_em
import os
from Real_Space_Gfunc import Sxx_Syy_from_G0


def main():
    # Constants and convention
    eps0 =epsilon_0
    mu0 = mu_0
    eta0 = mu_0*c
    nm = 1e-9

    # Particle parameter
    r = 65*nm # Single scatterer radius

    # wavelength grid (meters)
    lam = np.linspace(450, 750, 301)*nm

    # environment
    n_d, med = 1,"Air" # use Air with n_d = 1 and Glass with n_d = 1.45
    epsd = (n_d)**2

    # lattice Parameter
    Px, Py = 220*nm, 600*nm
    A = Px * Py
    
    R_list = []
    T_list = []
    Ab_list = []
    for wvl in lam:
        k0 = 2*np.pi / wvl
        kd = n_d * k0
        
        alpha_p1x, alpha_m1y = alpha_em(n_d, wvl, r, n=1, wv_scale=nm)

        ##___Green function methods, select one of them
        Sxx, Syy = Sxx_Syy_from_G0(kd=kd, Px=Px, Py=Py, N=200)
       
        ##___Effective polarizability
        alpha_peff = 1/((1/alpha_p1x)-Sxx)
        alpha_meff = 1/((1/alpha_m1y)-Syy)
        
        ##___Reflectance, transmittance and absorbance
        fac = 1j*kd/(2*A)
        r_amp= fac*(alpha_peff - alpha_meff)
        t_amp = 1 + fac*(alpha_peff + alpha_meff)
        R = abs(r_amp)**2
        T = abs(t_amp)**2
        Ab = 1-T-R
        
        R_list.append(R)
        T_list.append(T)
        Ab_list.append(Ab)
    plt.plot(lam/1e-9, R_list, label="Reflectance (R), Couple-Dipole")
    plt.plot(lam/1e-9, T_list, label="Transmittance (T), Couple-Dipole")
    plt.plot(lam/1e-9, Ab_list, label="Absorption (A), Couple-Dipole")
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("R, T, A")
    plt.legend()
    plt.show()

main()

