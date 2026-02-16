import numpy as np

def lattice_Sxx_Syy_realspace(kd, Dx, Dy=None, Nx=50, Ny=50, convention="exp(+ikR)"):
    if Dy is None:
        Dy = Dx
    if convention not in ("exp(-ikR)", "exp(+ikR)"):
        raise ValueError("convention must be 'exp(-ikR)' or 'exp(+ikR)'")

    sgn = -1.0 if convention == "exp(-ikR)" else +1.0

    sumGxx = 0.0 + 0.0j
    sumGyy = 0.0 + 0.0j

    for nx in range(-Nx, Nx + 1):
        x = nx * Dx
        for ny in range(-Ny, Ny + 1):
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

    Sxx = (kd * kd) * sumGxx
    Syy = (kd * kd) * sumGyy
    return Sxx, Syy

