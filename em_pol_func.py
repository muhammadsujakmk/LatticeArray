import numpy as np
import matplotlib.pyplot as plt
import recmethod as rc
import matimport as mat



def alpha_em(n_med, wv, r, n=1, wv_scale = 1e-9):
    epsd = (n_med)**2 
    mat_core=mat.mat_cal(wv/wv_scale,r"\Si_Green.txt")
    k0 = 2*n_med*np.pi/wv
    x1 = k0*r
    m1 = mat_core / n_med
    
    n=1 
    psi_n, psi_n1 = rc.ricabes1(n, x1)
    zeta_n, zeta_n1 = rc.ricabes3(n, x1)

    D = rc.Dn(m1*x1, n)
    print(D)
    a = ((D/m1 + n/x1) * psi_n - psi_n1) / ((D/m1+ n/x1) * zeta_n - zeta_n1)
    
    b = ((m1*D + n/x1) * psi_n - psi_n1) / ((m1*D + n/x1) * zeta_n - zeta_n1)
    #alpha_e = 1j*6*np.pi*eps0*epsd*a/(k0)**3  #Unit of SI (F m^2) 
    alpha_e = 1j*6*np.pi*a/(k0)**3 # Unit of volume (m^3)
    alpha_m = 1j*6*np.pi*b/(k0)**3
    
    return alpha_e, alpha_m



