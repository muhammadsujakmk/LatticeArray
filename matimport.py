import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate


def interpol(x,y,f):
    fnew = interpolate.splrep(x,y)
    return interpolate.splev(f,fnew)

def mat_cal(wvl,material):
    path = r'F:\101\RESEARCH MOL\Surface Lattice\All-Dielectric\Silicon nanodisk\Array Silicon Cal\SemiAnalytic Calculation\LatticeArray\material library'
    data = np.loadtxt(path+material)
    x = data[:,0]   #wavelength with nm unit 
    n = data[:,1]       # real RI
    k = data[:,2]       # Imag RI
   
    n_new = interpol(x,n,wvl)
    k_new = interpol(x,k,wvl)
    return n_new+1j*k_new# NOTE: "+" sign for loss and "-" for gain 
