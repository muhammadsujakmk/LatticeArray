import numpy as np

def _sqrt_branch(z):
    w = np.sqrt(z)
    if (w.real < 0) or (w.real == 0 and w.imag < 0):
        w = -w
    return w

def lattice_sum_sub(kd, Px, Py, zp, epsd, epss, N=20, convention="exp(+ikR)"):
    """
    ------Full reflected dyadic Green's function style------
    ------contructed from NF/MF/FF+Fresnel rs, rp-----------
    
    Lattice sum including substrate reflection with sign switching.

    convention:
      - "exp(+ikR)" uses exp(+ i k R)
      - "exp(-ikR)" uses exp(- i k R)
    """
    if convention not in ("exp(+ikR)", "exp(-ikR)"):
        raise ValueError("convention must be 'exp(+ikR)' or 'exp(-ikR)'")

    sgn = +1.0 if convention == "exp(+ikR)" else -1.0
    
    Sxx = 0.0 + 0.0j; Syy = 0.0 + 0.0j
    Srxx = 0.0 + 0.0j; Sryy = 0.0 + 0.0j
    for nx in range(-N, N+1):
        x = nx*Px
        for ny in range(-N, N+1):
            if nx ==0 and ny==0:
                continue
            
            y = ny*Py
            r = np.hypot(x,y)
            R = np.sqrt(r*r + zp*zp)
             
            exp_dir = kd**2/(4*np.pi) * np.exp(1j*sgn*kd*r) 
            exp_ref = kd**2/(4*np.pi) * np.exp(1j*sgn*kd*R) 
            phase = 1j*sgn*kd*zp**2/R

            ## Fresnel coef
            res1 = zp / R
            root = _sqrt_branch( (epss-epsd)/epsd + res1**2)
            num_rs = res1 - root
            den_rs = res1 + root
            rs = num_rs/den_rs 
            num_rp = epss*res1 - epsd*root
            den_rp = epss*res1 + epsd*root
            rp = num_rp/den_rp 
            
            ## FF terms
            gxx = (x*x)/(R*R)
            gyy = (y*y)/(R*R)
            gzz = (zp*zp)/(R*R)

            Sxx += exp_ref / R * np.exp(-phase) * (1 - gxx * (1-gzz))
            Syy += exp_ref / R * np.exp(-phase) * (1 - gyy * (1-gzz))
            
            ## R, FF terms
            Srxx += exp_ref / R * np.exp(phase) * (rs - gxx * (rs+rp*gzz))
            Sryy += exp_ref / R * np.exp(phase) * (rs - gyy * (rs+rp*gzz))
            
            ## MF terms
            gxx_ = (x*x)/(r*r)
            gyy_ = (y*y)/(r*r)
            Sxx += exp_dir / (r*r) * (1j/ (sgn*kd) - 3j*gxx_/(sgn*kd))
            Syy += exp_dir / (r*r) * (1j/ (sgn*kd) - 3j*gyy_/(sgn*kd))
            
            ## NF terms
            Sxx += exp_dir / r**3 * (-1/kd**2 + 3*gxx_/(kd**2))
            Syy += exp_dir / r**3 * (-1/kd**2 + 3*gyy_/(kd**2))
            
    return Sxx+Srxx, Syy+Sryy



def lattice_sum_sub_modif(kd, Px, Py, zp, epsd, epss, N=20, convention="exp(+ikR)"):
    """
    ------Substrate modifies ONLY the far-field lattice term-----
    with r_l^(s) from main-text Eq. (4), and S_l^FF from SI Eq. (S7a)
    IMPORTANT: R_l in the phase is the IN_PLANE distance r =sqrt(x^2+y^2)


    convention:
      - "exp(+ikR)" uses exp(+ i k R)
      - "exp(-ikR)" uses exp(- i k R)
    """
    if convention not in ("exp(+ikR)", "exp(-ikR)"):
        raise ValueError("convention must be 'exp(+ikR)' or 'exp(-ikR)'")

    sgn = +1.0 if convention == "exp(+ikR)" else -1.0
    
    Delta_eps = (epss - epsd)
    if np.isclose(Delta_eps,0.0):
        print("Two medium is nearly symmetric!")
        Rtilde = np.inf
    else:
        Rtilde = zp * np.sqrt( epsd / np.abs(Delta_eps) )

    Sxx = 0.0 + 0.0j; Syy = 0.0 + 0.0j
    for nx in range(-N, N+1):
        x = nx*Px
        for ny in range(-N, N+1):
            if nx ==0 and ny==0:
                continue
            
            y = ny*Py
            R = np.hypot(x,y)
             

            ## Fresnel coef main-text Eq. (4)
            if np.isinf(Rtilde):
                rs = 0
            else:
                root = np.sqrt( (R*R)/(Rtilde*Rtilde) + 1.0)
                num_rs = 1 - root
                den_rs = 1 + root
                rs = num_rs/den_rs 
            
            ## FF terms Eq. (S7a)
            exp_ref = kd**2/(4*np.pi) * np.exp(1j*sgn*kd*R) 
            gxx = (x*x)/(R*R)
            gyy = (y*y)/(R*R)

            Sxx += exp_ref / R * (1.0 - gxx) * (1.0 + rs)
            Syy += exp_ref / R * (1.0 - gyy) * (1.0 + rs)
            
    return Sxx, Syy


def lattice_sum(kd, Px, Py, N=20, convention="exp(+ikR)"):


    if convention not in ("exp(+ikR)", "exp(-ikR)"):
        raise ValueError("convention must be 'exp(+ikR)' or 'exp(-ikR)'")
    Sxx = 0.0 + 0.0j; Syy = 0.0 + 0.0j
    sgn = +1.0 if convention == "exp(+ikR)" else -1.0

    for nx in range(-N, N+1):
        x = nx*Px
        for ny in range(-N, N+1):
            if nx ==0 and ny==0:
                continue
             
            y = ny*Py
            R = np.hypot(x,y)
            res0 = kd**2/(4*np.pi) * np.exp(1j*sgn*kd*R) 
            
            ## FF terms
            gxx = (x*x)/(R*R)
            gyy = (y*y)/(R*R)
            Sxx += res0 / R * (1 - gxx)
            Syy += res0 / R * (1 - gyy)
            
            ## MF terms
            Sxx += res0 / R**2 * (1j/ (sgn*kd) - 3j*x*x /(sgn*kd*R**2))
            Syy += res0 / R**2 * (1j/ (sgn*kd) - 3j*y*y /(sgn*kd*R**2))
            
            ## NF terms
            Sxx += res0 / R**3 * (-1/kd**2 + 3*x**2/(kd**2 * R**2))
            Syy += res0 / R**3 * (-1/kd**2 + 3*y**2/(kd**2 * R**2))
            
    return Sxx, Syy




