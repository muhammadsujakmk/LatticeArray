import numpy as np

##________________________________________________________________________
## Python code for calculating array response in a homogeneous environment.
## The formulation of Green's Function in Real Space is adopted 
## from paper: Laser Photonics Rev. 2017, 11, 1700132
##______________________________________________________________

def lattice_Sxx_Syy_realspace(kd, Dx, Dy=None, N=50, convention="exp(+ikR)"):
    if Dy is None:
        Dy = Dx
    if convention not in ("exp(-ikR)", "exp(+ikR)"):
        raise ValueError("convention must be 'exp(-ikR)' or 'exp(+ikR)'")

    sgn = -1.0 if convention == "exp(-ikR)" else +1.0

    sumGxx = 0.0 + 0.0j
    sumGyy = 0.0 + 0.0j

    for nx in range(-N, N + 1):
        x = nx * Dx
        for ny in range(-N, N + 1):
            if nx == 0 and ny == 0:
                continue
            y = ny * Dy
            R = np.hypot(x, y)
            ex = x / R
            ey = y / R

            exp_ikR = np.exp(1j * sgn * kd * R)
            invR  = 1.0 / R
            invR2 = invR * invR
            invR3 = invR2 * invR

            a = invR + 1j*sgn*(1.0/kd)*invR2 - (1.0/(kd*kd))*invR3
            b = -invR - 3j*sgn*(1.0/kd)*invR2 + (3.0/(kd*kd))*invR3

            pref = exp_ikR / (4.0 * np.pi)
            sumGxx += pref * (a + b * ex * ex)
            sumGyy += pref * (a + b * ey * ey)

    Gxx = sumGxx
    Gyy = sumGyy
    return Gxx, Gyy

def Sxx_Syy_from_G0(kd, Px, Py, N):
    eta = 1e-3
    kdG = kd * (1+1j*eta)
    Gxx, Gyy = lattice_Sxx_Syy_realspace(kdG, Dx=Px, Dy=Py, N=N, convention = "exp(+ikR)")
    return (kd * kd)*Gxx, (kd * kd)*Gyy




