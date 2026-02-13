import recmethod as rc
import matimport as mat
import mpmath as mp

mp.mp.dps = 80

def safe_div(num, den, eps=mp.mpf('1e-60')):
    if abs(den) < eps:
        den = den + eps*(1+1j)
    return num/den


def alpha_em(n_med, wv, r, n=1, wv_scale = 1e-9):
    wv = mp.mpf(wv) 
    r = mp.mpf(r) 
    n_med = mp.mpf(n_med) 
    epsd = (n_med)**2 
    
    mat_core=mat.mat_cal(wv/wv_scale,r"\Si_Green.txt")
    k0 = 2*n_med*mp.pi/wv
    x1 = k0*r
    x1 = x1 + 1j*mp.mpf("1e-30")
    m1 = mp.mpc(mat_core) / n_med
    psi_n, psi_n1 = rc.ricabes1(n, x1)
    zeta_n, zeta_n1 = rc.ricabes3(n, x1)

    D = rc.Dn(m1*x1, n)
    a = ((D/m1 + n/x1) * psi_n - psi_n1) / ((D/m1+ n/x1) * zeta_n - zeta_n1)
    
    b = ((m1*D + n/x1) * psi_n - psi_n1) / ((m1*D + n/x1) * zeta_n - zeta_n1)
    #alpha_e = 1j*6*mp.pi*eps0*epsd*a/(k0)**3  #Unit of SI (F m^2) 
    alpha_e = 1j*6*mp.pi*a/(k0)**3 # Unit of volume (m^3)
    alpha_m = 1j*6*mp.pi*b/(k0)**3
    
    return alpha_e, alpha_m



