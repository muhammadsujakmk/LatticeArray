import numpy as np
from scipy.constants import epsilon_0, mu_0, c
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from em_pol_func import alpha_em
import os
from Reciprocal_Space_Gfun import lattice_S_from_prb
from Real_Space_TrunGreenFunc import lattice_Sxx_Syy_realspace



def path():
    return r"F:\101\RESEARCH MOL\Surface Lattice\All-Dielectric\Silicon nanodisk\Array Silicon Cal\SemiAnalytic Calculation\COMSOL_Calculation"


def Pol_single(wvl, med):
    E0 = 1 #V/m
    
    # Import polarizability normal/top from COMSOL 
    fileNormal = os.path.join(path(), f"Single_Scat_D200H100_EDxMDy_med{med}_normal")
    data_pxNormal = np.loadtxt(fileNormal+"_EDx_clean.txt", dtype=complex) 
    data_myNormal = np.loadtxt(fileNormal+"_MDy_clean.txt", dtype=complex) 
    
    wvl_grid = data_pxNormal[:,0]
    pxNormal_fun = interp1d(wvl_grid,data_pxNormal[:,1])
    myNormal_fun = interp1d(wvl_grid,data_myNormal[:,1])

    pxNormal_val = pxNormal_fun(wvl)
    myNormal_val = myNormal_fun(wvl)
    
    # Import polarizability lateral/side from COMSOL 
    fileLateral = os.path.join(path(), f"Single_Scat_D200H100_EDxMDy_med{med}_lateral")
    data_pxLateral = np.loadtxt(fileLateral+"_EDx_clean.txt", dtype=complex) 
    data_myLateral = np.loadtxt(fileLateral+"_MDy_clean.txt", dtype=complex) 
    
    wvl_grid = data_pxLateral[:,0]
    pxLateral_fun = interp1d(wvl_grid,data_pxLateral[:,1])
    myLateral_fun = interp1d(wvl_grid,data_myLateral[:,1])

    pxLateral_val = pxLateral_fun(wvl)
    myLateral_val = myLateral_fun(wvl)
    
    #return pxNormal_val.real+1j*pxNormal_val.imag, myNormal_val.real+1j*myNormal_val.imag,pxLateral_val.real+1j*pxLateral_val.imag, myLateral_val.real+1j*myLateral_val.imag
    return pxNormal_val, myNormal_val,pxLateral_val, myLateral_val


def lattice_sum(kd, P, N=20):
    Sxx = 0.0 + 0.0j; Syy = 0.0 + 0.0j
    for nx in range(-N, N+1):
        for ny in range(-N, N+1):
            if nx ==0 and ny==0:
                continue
             
            y = ny*P
            x = nx*P
            R = np.hypot(x,y)
            res0 = kd**2/(4*np.pi) * np.exp(1j*kd*R) 
            
            ## FF terms
            Sxx += res0 / R * (1 - x**2/R**2)
            Syy += res0 / R * (1 - y**2/R**2)
            
            ## MF terms
            Sxx += res0 / R**2 * (1j/kd - 3j*x**2/(kd*R**2))
            Syy += res0 / R**2 * (1j/kd - 3j*y**2/(kd*R**2))
            
            ## NF terms
            Sxx += res0 / R**3 * (-1/kd**2 + 3*x**2/(kd**2 * R**2))
            Syy += res0 / R**3 * (-1/kd**2 + 3*y**2/(kd**2 * R**2))
            
    return Sxx, Syy

def radiative_correction(alpha, kd):
    return 1.0 / (1.0/alpha - 1j*kd**3/(6*np.pi))


def main():
    eps0 =epsilon_0
    mu0 = mu_0
    eta0 = mu_0*c
    nm = 1e-9

    # Particle parameter
    r = 65*nm # Single scatterer radius

    # wavelength grid (meters)
    lam = np.linspace(450, 750, 1000)*nm

    # environment
    n_d, med = 1.45,"Glass" # use Air with n_d = 1 and Glass with n_d = 1.45
    epsd = (n_d)**2

    # lattice
    Px = 220*nm
    Py = 517*nm
    A = Px * Py
    
    R_list = []
    T_list = []
    Ab_list = []
    S_list = []
    inv_ap_list = []
    inv_am_list= []
    for wvl in lam: 
        k0 = 2*np.pi / wvl
        kd = n_d * k0 
        
        #----Polarizabilty of single scatterer
        #alpha_p1x, alpha_m1y, alpha_p2x, alpha_m2y = Pol_single(wvl/1e-9, med)
        alpha_p1x, alpha_m1y = alpha_em(n_d, wvl, r, n=1, wv_scale=nm)
        #alpha_p1x = radiative_correction(alpha_p1x, kd)
        #alpha_p2x = radiative_correction(alpha_p2x, kd)

        #alpha_m1y = radiative_correction(alpha_m1y, kd)
        #alpha_m1y = radiative_correction(alpha_m1y, kd)

        ##___Green function methods, select one of them
        #Sxx, Syy = lattice_S_from_prb(kd, P, N=51, convention = "exp(+ikR)")
        Sxx, Syy = lattice_Sxx_Syy_realspace(kd, Dx=Px, Dy=Py, N=51, convention = "exp(+ikR)")
        #Sxx, Syy = lattice_sum(kd, P, N=151)
        #Sxx_list.append(Sxx)
        #Syy_list.append(Syy)
        
        inv_alpha_peff = (1/alpha_p1x) - Sxx
        #inv_alpha_peff = (1/alpha_p1x) - (alpha_p2x/alpha_p1x)*Sxx
        #inv_alpha_peff = ((1/alpha_p1x) - Sxx)
        inv_alpha_meff = (1/alpha_m1y) - Syy
        #inv_alpha_meff = (1/alpha_m1y) - (alpha_m2y/alpha_m1y)*Syy
        #inv_alpha_meff = ((1/alpha_m1y) - Syy)
        alpha_peff = 1/inv_alpha_peff
        alpha_meff = 1/inv_alpha_meff
        
        inv_ap_list.append(inv_alpha_peff)     # complex
        inv_am_list.append(inv_alpha_meff)     # complex

        
        fac = -1j*kd/(2*A)
        r_ = fac*(alpha_peff - alpha_meff)
        t_ = 1 - fac*(alpha_peff + alpha_meff)
        R = abs(r_)**2
        T = abs(t_)**2
        Ab = 1-T-R
        """
        R = fac**2 * ( (np.real(alpha_peff)-np.real(alpha_meff))**2 + (np.imag(alpha_peff)-np.imag(alpha_meff))**2 )
        T = ( 1-fac*((np.imag(alpha_peff)-np.imag(alpha_meff))) )**2 + fac**2 * (np.real(alpha_peff)-np.real(alpha_meff))**2 
        Ab = 1-R-T 
        """
        R_list.append(R)
        T_list.append(T)
        Ab_list.append(Ab)
        #print("Imag(alpha_p1x) = ", 1/np.imag(alpha_p1x))
        #print("R = ", R) 
        #print("--------------------") 
    plt.plot(lam/1e-9, R_list, label="Reflectance (R), Couple-Dipole")
    plt.plot(lam/1e-9, T_list, label="Transmittance (T), Couple-Dipole")
    #plt.plot(lam/1e-9, Ab_list, label="Absorption (A), Couple-Dipole")
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("R, T, A")
    plt.legend()
    plt.show()

main()

