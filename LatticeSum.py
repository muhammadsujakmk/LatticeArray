import numpy as np

def lattice_sum_sub(kd, Px, Py, zp, epsd, epss, N=20):
    Sxx = 0.0 + 0.0j; Syy = 0.0 + 0.0j
    Srxx = 0.0 + 0.0j; Sryy = 0.0 + 0.0j
    for nx in range(-N, N+1):
        x = nx*Px
        for ny in range(-N, N+1):
            if nx ==0 and ny==0:
                continue
            
            y = ny*Py
            r = np.sqrt(x**2 + y**2)
            R = np.sqrt(r**2 + zp**2)
             
            exp_dir = kd**2/(4*np.pi) * np.exp(1j*kd*r) 
            exp_ref = kd**2/(4*np.pi) * np.exp(1j*kd*R) 
            phase = 1j*kd*zp**2/R

            ## Fresnel coef
            res1 = zp / R
            root = np.sqrt( (epss-epsd)/epsd + res1**2)
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
            Sxx += exp_dir / r**2 * (1j/kd - 3j*gxx_/(kd))
            Syy += exp_dir / r**2 * (1j/kd - 3j*gyy_/(kd))
            
            ## NF terms
            Sxx += exp_dir / r**3 * (-1/kd**2 + 3*gxx_/(kd**2))
            Syy += exp_dir / r**3 * (-1/kd**2 + 3*gyy_/(kd**2))
            
    return Sxx+Srxx, Syy+Sryy

def lattice_sum(kd, Px, Py, N=20):
    Sxx = 0.0 + 0.0j; Syy = 0.0 + 0.0j
    for nx in range(-N, N+1):
        x = nx*Px
        for ny in range(-N, N+1):
            if nx ==0 and ny==0:
                continue
             
            y = ny*Py
            R = np.hypot(x,y)
            res0 = kd**2/(4*np.pi) * np.exp(1j*kd*R) 
            
            ## FF terms
            gxx = (x*x)/(R*R)
            gyy = (y*y)/(R*R)
            Sxx += res0 / R * (1 - gxx)
            Syy += res0 / R * (1 - gyy)
            
            ## MF terms
            Sxx += res0 / R**2 * (1j/kd - 3j*x**2/(kd*R**2))
            Syy += res0 / R**2 * (1j/kd - 3j*y**2/(kd*R**2))
            
            ## NF terms
            Sxx += res0 / R**3 * (-1/kd**2 + 3*x**2/(kd**2 * R**2))
            Syy += res0 / R**3 * (-1/kd**2 + 3*y**2/(kd**2 * R**2))
            
    return Sxx, Syy



