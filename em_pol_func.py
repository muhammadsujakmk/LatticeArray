import recmethod as rc
import matimport as mat
import mpmath as mp

mp.mp.dps = 80

def alpha_em(n_med, wv, r, n=1, wv_scale = 1e-9):
    wv = mp.mpf(wv) 
    r = mp.mpf(r) 
    n_med = mp.mpf(n_med) 
    epsd = (n_med)**2
    eps0 = 8.854e-12 # Unit of F/m
    
    mat_core=mat.mat_cal(wv/wv_scale,r"\Si_Green.txt")
    k0 = 2*mp.pi/wv
    kd = k0*n_med
    x1 = kd*r
    m1 = mp.mpc(mat_core) / n_med
    psi_n, psi_n1 = rc.ricabes1(n, x1)
    zeta_n, zeta_n1 = rc.ricabes3(n, x1)

    D = rc.Dn(m1*x1, n)
    a = ((D/m1 + n/x1) * psi_n - psi_n1) / ((D/m1+ n/x1) * zeta_n - zeta_n1)
    
    b = ((m1*D + n/x1) * psi_n - psi_n1) / ((m1*D + n/x1) * zeta_n - zeta_n1)
    #alpha_e = 1j*6*mp.pi*eps0*epsd*a/(kd)**3  #Unit of SI (F m^2) 
    alpha_e = 1j*6*mp.pi*a/(kd)**3 # Unit of volume (1 / m^3)
    alpha_m = 1j*6*mp.pi*b/(kd)**3
    return alpha_e, alpha_m



