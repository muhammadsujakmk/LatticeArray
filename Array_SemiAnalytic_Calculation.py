import numpy as np
from scipy.constants import epsilon_0, mu_0, c
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

def path():
    return r"C:\RESEARCH\RESEARCH MOL\Surface Lattice\All-Dielectric\Silicon nanodisk\SemiAnalytic Calculation\COMSOL_Calculation"

def Pol_single(wvl):
    eps0 = epsilon_0
    zeta0 = mu_0*c
    E0 = 1 #V/m
    
    # Import polarizability normal/top from COMSOL 
    filename=path()+"\Single_Scat_D200H100_EDxMDy_medAir_normal"
    data_pxNormal = np.loadtxt(filename+"_EDx_clean.txt", dtype=complex) 
    data_myNormal = np.loadtxt(filename+"_MDy_clean.txt", dtype=complex) 
    
    wvl_grid = data_pxNormal[:,0].real
    pxNormal_fun = interp1d(wvl_grid,data_pxNormal[:,1])
    myNormal_fun = interp1d(wvl_grid,data_myNormal[:,1])

    pxNormal_val = pxNormal_fun(wvl)/E0
    myNormal_val = myNormal_fun(wvl)*zeta0/E0
    
    # Import polarizability lateral/side from COMSOL 
    filename=path()+"\Single_Scat_D200H100_EDxMDy_medAir_lateral"
    data_pxLateral = np.loadtxt(filename+"_EDx_clean.txt", dtype=complex) 
    data_myLateral = np.loadtxt(filename+"_MDy_clean.txt", dtype=complex) 
    
    wvl_grid = data_pxLateral[:,0].real
    pxLateral_fun = interp1d(wvl_grid,data_pxLateral[:,1])
    myLateral_fun = interp1d(wvl_grid,data_myLateral[:,1])

    pxLateral_val = pxLateral_fun(wvl)/E0
    myLateral_val = myLateral_fun(wvl)*zeta0/E0
    
    return -pxNormal_val.real+1j*pxNormal_val.imag, -myNormal_val.real+1j*myNormal_val.imag,-pxLateral_val.real+1j*pxLateral_val.imag, -myLateral_val.real+1j*myLateral_val.imag 


def lattice_sum(kd, P, N=20):
    SFF = 0.0 + 0.0j
    SMF = 0.0 + 0.0j
    SNF = 0.0 + 0.0j
    for nx in range(-N, N+1):
        for ny in range(-N, N+1):
            if nx ==0 and ny==0:
                continue
            R = P * np.sqrt(nx**2 + ny**2)
            x = nx*P
            SFF += kd**2/(4*np.pi) * np.exp(1j*kd*R)/R * (1 - x**2/R**2)
            SMF += kd**2/(4*np.pi) * np.exp(1j*kd*R)/R**2 * (1j/kd - 3j*x**2/(kd*R**2))
            SNF += kd**2/(4*np.pi) * np.exp(1j*kd*R)/R**3 * (-1/kd**2 + 3*x**2/(kd**2 * R**2))
            
    return SFF+SMF+SNF


def main():
    eps0 =epsilon_0 

    # wavelength grid (meters)
    lam = np.linspace(600e-9, 1000e-9, 1001)
    omega = 2*np.pi*c / lam 

    # environment
    n_d = 1.45               # air
    eps_d = n_d**2

    # lattice
    P = 550e-9
    A = P**2
    
    R_list = []
    T_list = []
    S_list = []
    for wvl in lam: 
        alpha_p, alpha_m = Pol_single(wvl/1e-9)
        k0 = 2*np.pi / wvl
        k_d = n_d * k0
        S = lattice_sum(k_d, P, N=100)
        
        #alpha_p_eff = 1 / (1/alpha_p - S)
        #alpha_m_eff = 1 / (1/alpha_m - S)
        
        S_list.append(S)


        #R = np.abs( (1j*kd /2*A) * (alpha_p_eff - alpha_m_eff) )**2
        #T = np.abs( 1-(1j*kd /2*A) * (alpha_p_eff + alpha_m_eff) )**2
    plt.plot(lam/1e-9, np.real(S_list), label="Real part of lattice sum")
    plt.plot(lam/1e-9, np.imag(S_list), label="Imaginary part of lattice sum")
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Lattice sum")
    plt.legend()
    plt.show()


main()

