import numpy as np

def lattice_S_from_prb(kd, P, N, convention="exp(+ikR)"):
    # compute sum of Gxx,Gyy (units 1/m)
    sumGxx = 0+0j
    sumGyy = 0+0j
    if convention=="exp(+ikR)":
        sign = +1
    elif convention=="exp(-ikR)":
        sign = -1
    else:
        raise ValueError("convention must be exp(+ikR) or exp(-ikR)")

    for nx in range(-N, N+1):
        x = nx*P
        for ny in range(-N, N+1):
            if nx==0 and ny==0: 
                continue
            y = ny*P
            R = np.hypot(x,y)
            ex, ey = x/R, y/R

            exp_ikR = np.exp(1j*sign*kd*R)
            invR, invR2, invR3 = 1/R, 1/R**2, 1/R**3

            a = invR + 1j*(1/kd)*invR2 - (1/(kd*kd))*invR3
            b = -invR - 1j*(3/kd)*invR2 + (3/(kd*kd))*invR3

            pref = exp_ikR/(4*np.pi)
            Gxx = pref*(a + b*ex*ex)
            Gyy = pref*(a + b*ey*ey)

            sumGxx += Gxx
            sumGyy += Gyy

    # Convert to S (units 1/m^3): multiply ONCE by kd^2
    Sxx = kd*kd*sumGxx
    Syy = kd*kd*sumGyy
    return Sxx, Syy



