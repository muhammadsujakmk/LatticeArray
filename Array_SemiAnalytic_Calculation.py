import numpy as np
from scipy.constants import epsilon_0, mu_0, c
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from em_pol_func import alpha_em
import os
from Real_Space_Gfunc import Sxx_Syy_from_G0
from LatticeSum import lattice_sum_sub, lattice_sum


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
    lam = np.linspace(450, 750, 61)*nm

    # environment
    n_d, med = 1,"Air" # use Air with n_d = 1 and Glass with n_d = 1.45
    epsd = (n_d)**2
    
    n_s, med_sub = 1.5, "Glass" # Substrate parameter
    epss = (n_s)**2
    zp = r # The distance of particle's center to substrate surface
    

    # lattice
    Px = 220*nm
    Py = 600*nm
    A = Px * Py
    
    R_list = []
    T_list = []
    Ab_list = []
    for wvl in lam:
        eta = 1e-3 
        k0 = 2*np.pi / wvl
        kd = n_d * k0 
        kdG = n_d * k0 * (1+1j*eta)
        
        #----Polarizabilty of single scatterer
        #alpha_p1x, alpha_m1y, alpha_p2x, alpha_m2y = Pol_single(wvl/1e-9, med)
        alpha_p1x, alpha_m1y = alpha_em(n_d, wvl, r, n=1, wv_scale=nm)
        #alpha_p1x = radiative_correction(alpha_p1x, kd)
        #alpha_p2x = radiative_correction(alpha_p2x, kd)

        #alpha_m1y = radiative_correction(alpha_m1y, kd)
        #alpha_m1y = radiative_correction(alpha_m1y, kd)

        ##___Green function methods, select one of them
        #Sxx, Syy = Sxx_Syy_from_G0(kd=kd, Dx=Px, Dy=Py, N=200)
        Sxx, Syy = lattice_sum(kdG, Px=Px, Py=Py, N=200)
        #Sxx, Syy = lattice_sum_sub(kdG, Px, Py, zp, epsd, epss, N=200)
       
        #_____Effective polarizabilty for homogeneous env_______
        #alpha_peff = 1/((1/alpha_p1x) - (alpha_p2x/alpha_p1x)*Syy)
        alpha_peff = 1/((1/alpha_p1x) - Sxx)
        #alpha_meff = 1/((1/alpha_m1y) - (alpha_m2y/alpha_m1y)*Syy)
        alpha_meff = 1/((1/alpha_m1y) - Syy)
            
        #_____with substrate effect________________________________
        C =  1/( 4*np.pi*(2*zp)**3 ) * ((epsd-epss)/(epsd+epss))
        inv_alpha_p1x_s = (1/alpha_p1x) + C
        #alpha_peff = 1/(inv_alpha_p1x_s - Sxx)
        #alpha_meff = 1/((1/alpha_m1y) - Syy)
        
        fac = 1j*kd/(2*A) # "-fac..." for substrate otherwise use "+fac..."
        r_amp= fac*(alpha_peff - alpha_meff)
        t_amp = 1 + fac*(alpha_peff + alpha_meff)
        #t_amp = 1 - fac*(alpha_peff + alpha_meff)  #__ for SI notation
        R = abs(r_amp)**2
        T = abs(t_amp)**2
        Ab = 1-T-R
        R_list.append(R)
        T_list.append(T)
        Ab_list.append(Ab)
    plt.plot(lam/1e-9, R_list, label="Reflectance (R), Couple-Dipole")
    plt.plot(lam/1e-9, T_list, label="Transmittance (T), Couple-Dipole")
    #plt.plot(lam/1e-9, Ab_list, label="Absorption (A), Couple-Dipole")
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("R, T, A")
    plt.legend()
    plt.show()

main()
