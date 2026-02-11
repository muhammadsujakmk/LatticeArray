import numpy as np
from scipy.constants import epsilon_0, mu_0, c
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import os

def path():
    return r"F:\101\RESEARCH MOL\Surface Lattice\All-Dielectric\Silicon nanodisk\Array Silicon Cal\SemiAnalytic Calculation\COMSOL_Calculation"


def Pol_single(wvl, med):
    E0 = 1 #V/m
    
    # Import polarizability normal/top from COMSOL 
    fileNormal = os.path.join(path(), f"Single_Scat_D200H100_EDxMDy_med{med}_normal")
    data_pxNormal = np.loadtxt(fileNormal+"_EDx_clean.txt", dtype=complex) 
    data_myNormal = np.loadtxt(fileNormal+"_MDy_clean.txt", dtype=complex) 
    
    wvl_grid = data_pxNormal[:,0].real
    pxNormal_fun = interp1d(wvl_grid,data_pxNormal[:,1])
    myNormal_fun = interp1d(wvl_grid,data_myNormal[:,1])

    pxNormal_val = pxNormal_fun(wvl)/E0
    myNormal_val = myNormal_fun(wvl)/E0
    
    # Import polarizability lateral/side from COMSOL 
    fileLateral = os.path.join(path(), f"Single_Scat_D200H100_EDxMDy_med{med}_lateral")
    data_pxLateral = np.loadtxt(fileLateral+"_EDx_clean.txt", dtype=complex) 
    data_myLateral = np.loadtxt(fileLateral+"_MDy_clean.txt", dtype=complex) 
    
    wvl_grid = data_pxLateral[:,0].real
    pxLateral_fun = interp1d(wvl_grid,data_pxLateral[:,1])
    myLateral_fun = interp1d(wvl_grid,data_myLateral[:,1])

    pxLateral_val = pxLateral_fun(wvl)/E0
    myLateral_val = myLateral_fun(wvl)/E0
    
    return pxNormal_val.real+1j*pxNormal_val.imag, myNormal_val.real+1j*myNormal_val.imag,pxLateral_val.real+1j*pxLateral_val.imag, myLateral_val.real+1j*myLateral_val.imag


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

def radiative_correction(alpha, kd):
    return 1.0 / (1.0/alpha - 1j*kd**3/(6*np.pi))


def main():
    eps0 =epsilon_0 

    # wavelength grid (meters)
    lam = np.linspace(601e-9, 999e-9, 1001)
    omega = 2*np.pi*c / lam 

    # environment
    n_d, med = 1.45,"Glass" # use Air with n_d = 1 and Glass with n_d = 1.45
    epsd = (n_d)**2

    # lattice
    P = 550e-9
    A = P**2
    
    R_list = []
    T_list = []
    Ab_list = []
    S_list = []
    inv_ap_list = []
    inv_am_list= []
    for wvl in lam: 
        k0 = 2*np.pi / wvl
        kd = n_d * k0
        
        
        alpha_p1x, alpha_m1y, alpha_p2x, alpha_m2y = Pol_single(wvl/1e-9, med)
        
        alpha_p1x = radiative_correction(alpha_p1x, kd)
        alpha_p2x = radiative_correction(alpha_p2x, kd)

        alpha_m1y = radiative_correction(alpha_m1y, kd)
        alpha_m2y = radiative_correction(alpha_m2y, kd)
        
        S = lattice_sum(kd, P, N=51)
        S_list.append(S)
        
        inv_alpha_peff = (1/alpha_p1x) - (alpha_p2x/alpha_p1x)*S
        inv_alpha_meff = (1/alpha_m1y) - (alpha_m2y/alpha_m1y)*S
        alpha_peff = 1/inv_alpha_peff
        alpha_meff = 1/inv_alpha_meff
        inv_ap_list.append(inv_alpha_peff)     # complex
        inv_am_list.append(inv_alpha_meff)     # complex


        R = np.abs( -1j*kd /(2*A) * (alpha_peff) )**2
        T = np.abs( 1+(1j*kd /(2*A)) * (alpha_peff) )**2
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

